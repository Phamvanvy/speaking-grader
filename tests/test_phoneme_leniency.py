"""Tests cho các lớp leniency mới (fix chấm chặt trên free-speech/video Band-9):

  - Fix A: assess_asr_confidence — gate free-speech theo Whisper word probability
    (mapping THUẦN tokenization, deterministic, không difflib).
  - Fix B: word_to_ipa_with_stress_source (nguồn IPA per-word) + clitic 's +
    cap penalty g2p_uncertain cho từ eSpeak-sourced.
  - Fix C: connected-speech elision — nuốt stop cuối từ trước phụ âm là phát âm
    bản xứ hợp lệ, không tính lỗi.
  - Learner-regression guards: lỗi học viên thật (nuốt /s/ cuối, th→t, nuốt stop
    trước nguyên âm) KHÔNG bị các fix mới nuốt mất — chốt bằng test, không bằng mắt.
"""

from src.phoneme.ipa import word_to_ipa_with_stress, word_to_ipa_with_stress_source
from src.phoneme.l1_vietnamese import PenaltyReason
from src.phoneme.models import PhonemePoint, PhonemeSegment, WordSpan
from src.phoneme.reliability import SkipReason, assess_asr_confidence
from src.phoneme.scoring import _apply_connected_speech, compute_phoneme_score


def _segs(phs, conf=0.9):
    return [
        PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=conf)
        for i, p in enumerate(phs)
    ]


# ── Fix A: ASR word-confidence gate (free-speech) ────────────────────────────

class TestAssessAsrConfidence:
    """Gate free-speech theo Whisper word probability — mapping thuần tokenization."""

    def test_low_prob_word_skipped(self):
        skips = assess_asr_confidence(
            ["hello", "world"], [("hello", 0.9), ("world", 0.3)],
            min_probability=0.55,
        )
        assert set(skips) == {1}
        assert skips[1].reason == SkipReason.ASR_LOW_CONFIDENCE

    def test_zero_prob_never_skipped(self):
        # prob <= 0 = "không có số liệu" (whisperx thiếu score / IFW luôn 0) → không skip.
        skips = assess_asr_confidence(
            ["hello", "world"], [("hello", 0.0), ("world", 0.0)],
            min_probability=0.55,
        )
        assert skips == {}

    def test_disabled_when_min_probability_zero(self):
        assert assess_asr_confidence(
            ["hello"], [("hello", 0.1)], min_probability=0.0,
        ) == {}

    def test_multi_token_whisper_word_inherits_prob(self):
        # 1 Whisper word "aehelp.com's" sinh 2 token → cả 2 index nhận cùng prob.
        skips = assess_asr_confidence(
            ["aehelp", "com's", "test"],
            [("aehelp.com's", 0.2), ("test", 0.9)],
            min_probability=0.55,
        )
        assert set(skips) == {0, 1}

    def test_occurrence_keying_with_repeated_words(self):
        skips = assess_asr_confidence(
            ["the", "cat", "the"],
            [("the", 0.9), ("cat", 0.9), ("the", 0.2)],
            min_probability=0.55,
        )
        assert set(skips) == {2}

    def test_dropped_reference_word_skipped_over(self):
        # Từ không tra được IPA bị drop khỏi spans → reference_words là SUBSEQUENCE
        # của token transcript; con trỏ phải nhảy qua token thừa mà không lệch index.
        skips = assess_asr_confidence(
            ["the", "fox"],
            [("the", 0.9), ("zzzz", 0.2), ("fox", 0.3)],
            min_probability=0.55,
        )
        assert set(skips) == {1}  # "fox" (index 1) prob 0.3; "zzzz" bị nhảy qua

    def test_split_whisper_words_merge_into_one_token(self):
        # Whisper tách "o'clock" thành 2 word ("o'" + "clock") nhưng transcript text
        # giữ nguyên → token "o'clock" phủ cả 2 word, prob = MIN của 2 word.
        skips = assess_asr_confidence(
            ["at", "o'clock"],
            [("at", 0.9), ("o'", 0.3), ("clock", 0.8)],
            min_probability=0.55,
            transcript_text="at o'clock",
        )
        assert set(skips) == {1}
        assert skips[1].reason == SkipReason.ASR_LOW_CONFIDENCE

    def test_whisper_word_missing_from_text_stays_unknown(self):
        # Word không tìm thấy trong text → ký tự không được phủ → unknown → không skip.
        skips = assess_asr_confidence(
            ["hello", "world"],
            [("xyz", 0.1), ("world", 0.9)],
            min_probability=0.55,
            transcript_text="hello world",
        )
        assert skips == {}

    def test_structural_mismatch_returns_empty(self):
        # Reference word không có trong token transcript → không gate (conservative).
        assert assess_asr_confidence(
            ["qqq"], [("abc", 0.1)], min_probability=0.55,
        ) == {}

    def test_punctuation_only_whisper_word_ignored(self):
        # Whisper word chỉ có dấu câu → không sinh token, không lệch mapping.
        skips = assess_asr_confidence(
            ["hello", "world"],
            [("hello,", 0.9), ("...", 0.5), ("world.", 0.2)],
            min_probability=0.55,
        )
        assert set(skips) == {1}


# ── Fix B: nguồn IPA (override/cmudict/espeak) + clitic 's ──────────────────

class TestG2PSource:
    """word_to_ipa_with_stress_source trả đúng nguồn từng layer + clitic 's."""

    def test_override_source(self):
        _sym, _st, source = word_to_ipa_with_stress_source("especially")
        assert source == "override"

    def test_cmudict_source(self):
        sym, _st, source = word_to_ipa_with_stress_source("cat")
        assert source == "cmudict" and sym

    def test_failed_source_for_empty(self):
        assert word_to_ipa_with_stress_source("") == ([], [], "failed")

    def test_wrapper_matches_source_variant(self):
        assert word_to_ipa_with_stress("cat") == word_to_ipa_with_stress_source("cat")[:2]

    def _patch_cmudict(self, monkeypatch, data):
        import src.phoneme.ipa as ipa_mod
        monkeypatch.setattr(ipa_mod, "_lookup_cmudict", lambda w: data.get(w))

    def test_clitic_voiceless_stem_appends_s(self, monkeypatch):
        self._patch_cmudict(monkeypatch, {"blork": ["B", "L", "AO1", "R", "K"]})
        sym, st, source = word_to_ipa_with_stress_source("blork's")
        assert source == "cmudict"
        assert sym[-1] == "s" and sym[:-1] == ["b", "l", "ɔː", "r", "k"]
        assert len(sym) == len(st)

    def test_clitic_sibilant_stem_appends_iz(self, monkeypatch):
        self._patch_cmudict(monkeypatch, {"bloss": ["B", "L", "AO1", "S"]})
        sym, _st, source = word_to_ipa_with_stress_source("bloss's")
        assert source == "cmudict"
        assert sym[-2:] == ["ɪ", "z"]

    def test_clitic_voiced_stem_appends_z(self, monkeypatch):
        self._patch_cmudict(monkeypatch, {"blom": ["B", "L", "AA1", "M"]})
        sym, _st, source = word_to_ipa_with_stress_source("blom's")
        assert source == "cmudict"
        assert sym[-1] == "z"

    def test_spans_carry_source(self, monkeypatch):
        import src.phoneme.ipa as ipa_mod

        real = ipa_mod.word_to_ipa_with_stress_source

        def fake(word):
            if word.lower() == "zorgle":
                return ["z", "ɔː", "ɡ"], [None, None, None], "espeak"
            return real(word)

        monkeypatch.setattr(ipa_mod, "word_to_ipa_with_stress_source", fake)
        _ph, spans, _st, _ds = ipa_mod.text_to_ipa_sequence_with_spans("the zorgle cat")
        assert [s.source for s in spans] == ["cmudict", "espeak", "cmudict"]


class TestG2PUncertainScoring:
    """Sub/del trên từ eSpeak-sourced bị cap penalty → severity low (hidden-noise)."""

    def test_sub_on_espeak_span_capped_low(self):
        ref = ["z", "ɔː", "l"]
        spans = [WordSpan("zorgle", 0, 3, "espeak")]
        score = compute_phoneme_score(_segs(["b", "ɔː", "l"], conf=0.95), ref, spans)
        point = score.words[0].phonemes[0]
        assert point.status == "sub"
        assert point.severity == "low"
        assert point.penalty_reason == PenaltyReason.G2P_UNCERTAIN.value

    def test_del_on_espeak_span_capped_low(self):
        # Thiếu /l/ cuối (coda) → deletion sạch; bình thường medium, espeak → cap low.
        ref = ["z", "ɔː", "l"]
        spans = [WordSpan("zorgle", 0, 3, "espeak")]
        score = compute_phoneme_score(_segs(["z", "ɔː"], conf=0.95), ref, spans)
        point = score.words[0].phonemes[2]
        assert point.status == "del"
        assert point.severity == "low"
        assert point.penalty_reason == PenaltyReason.G2P_UNCERTAIN.value

    def test_cmudict_span_not_capped(self):
        ref = ["z", "ɔː", "l"]
        spans = [WordSpan("zorgle", 0, 3, "cmudict")]
        score = compute_phoneme_score(_segs(["b", "ɔː", "l"], conf=0.95), ref, spans)
        point = score.words[0].phonemes[0]
        assert point.status == "sub"
        assert point.severity in ("medium", "high")


# ── Fix C: connected-speech elision (nuốt stop cuối từ khi nối từ) ───────────

class TestApplyConnectedSpeech:
    """Post-pass unit-level: điều kiện flip chính xác, không phụ thuộc DTW."""

    # ref "test prep": t e s t | p r e p
    REF = ["t", "e", "s", "t", "p", "r", "e", "p"]
    SPANS = [WordSpan("test", 0, 4), WordSpan("prep", 4, 8)]

    def _ok(self, sym):
        return (PhonemePoint(symbol=sym, status="ok"), 0.0)

    def _result_all_ok(self):
        return {i: self._ok(s) for i, s in enumerate(self.REF)}

    def test_c1_final_stop_deletion_flipped(self):
        result = self._result_all_ok()
        result[3] = (PhonemePoint(symbol="t", status="del", severity="medium"), 0.5)
        raw = {3: 0.5}
        _apply_connected_speech(result, raw, self.REF, self.SPANS, [False] * 8)
        point, pen = result[3]
        assert point.status == "ok"
        assert point.penalty_reason == PenaltyReason.CONNECTED_SPEECH.value
        assert pen == 0.0 and 3 not in raw

    def test_c2_c3_smeared_sub_and_onset_deletion_flipped(self):
        # test /tesp/: sub t→p trên "test" + del Ø→p trên "prep(aration)".
        result = self._result_all_ok()
        result[3] = (PhonemePoint(symbol="t", status="sub", heard="p", severity="high"), 0.9)
        result[4] = (PhonemePoint(symbol="p", status="del", severity="high"), 0.9)
        raw = {3: 0.9, 4: 0.9}
        _apply_connected_speech(result, raw, self.REF, self.SPANS, [False] * 8)
        assert result[3][0].status == "ok"
        assert result[4][0].status == "ok"
        assert result[4][0].penalty_reason == PenaltyReason.CONNECTED_SPEECH.value
        assert raw == {}

    def test_c3_never_stands_alone(self):
        # Onset del của từ kế mà KHÔNG kèm sub smear ở từ trước → giữ nguyên lỗi.
        result = self._result_all_ok()
        result[4] = (PhonemePoint(symbol="p", status="del", severity="high"), 0.9)
        _apply_connected_speech(result, {4: 0.9}, self.REF, self.SPANS, [False] * 8)
        assert result[4][0].status == "del"

    def test_sub_heard_not_next_onset_kept(self):
        # sub t→k (k KHÔNG phải onset của "prep") → lỗi thật, giữ nguyên.
        result = self._result_all_ok()
        result[3] = (PhonemePoint(symbol="t", status="sub", heard="k", severity="high"), 0.9)
        _apply_connected_speech(result, {3: 0.9}, self.REF, self.SPANS, [False] * 8)
        assert result[3][0].status == "sub"

    def test_non_stop_final_not_flipped(self):
        # "nice day": /s/ cuối bị nuốt — lỗi L1 VN kinh điển, KHÔNG được tha.
        ref = ["n", "aɪ", "s", "d", "eɪ"]
        spans = [WordSpan("nice", 0, 3), WordSpan("day", 3, 5)]
        result = {i: self._ok(s) for i, s in enumerate(ref)}
        result[2] = (PhonemePoint(symbol="s", status="del", severity="medium"), 0.5)
        _apply_connected_speech(result, {2: 0.5}, ref, spans, [False] * 5)
        assert result[2][0].status == "del"

    def test_vowel_onset_next_word_not_flipped(self):
        # "test out": từ kế mở đầu NGUYÊN ÂM → phải nối âm, nuốt /t/ là lỗi.
        ref = ["t", "e", "s", "t", "aʊ", "t"]
        spans = [WordSpan("test", 0, 4), WordSpan("out", 4, 6)]
        result = {i: self._ok(s) for i, s in enumerate(ref)}
        result[3] = (PhonemePoint(symbol="t", status="del", severity="medium"), 0.5)
        _apply_connected_speech(result, {3: 0.5}, ref, spans, [False] * 6)
        assert result[3][0].status == "del"

    def test_mid_word_deletion_not_flipped(self):
        # Nuốt /s/ GIỮA từ "test" → không phải biên từ, giữ nguyên.
        result = self._result_all_ok()
        result[2] = (PhonemePoint(symbol="s", status="del", severity="medium"), 0.5)
        _apply_connected_speech(result, {2: 0.5}, self.REF, self.SPANS, [False] * 8)
        assert result[2][0].status == "del"

    def test_skipped_word_untouched(self):
        result = self._result_all_ok()
        result[3] = (PhonemePoint(symbol="t", status="del", severity="medium"), 0.5)
        skipped = [True] * 4 + [False] * 4
        _apply_connected_speech(result, {3: 0.5}, self.REF, self.SPANS, skipped)
        assert result[3][0].status == "del"


class TestConnectedSpeechEndToEnd:
    """compute_phoneme_score end-to-end: elision được tha, flag off = hành vi cũ."""

    REF = ["t", "e", "s", "t", "p", "r", "e", "p"]
    SPANS = [WordSpan("test", 0, 4), WordSpan("prep", 4, 8)]

    def test_elided_final_t_scores_perfect(self):
        # Nói /tes-prep/ (nuốt t cuối "test") → không lỗi medium/high, accuracy 1.0.
        score = compute_phoneme_score(
            _segs(["t", "e", "s", "p", "r", "e", "p"]), self.REF, self.SPANS,
        )
        assert not [e for e in score.errors if e.severity in ("medium", "high")]
        assert score.overall_accuracy == 1.0
        reasons = [p.penalty_reason for w in score.words for p in w.phonemes]
        assert PenaltyReason.CONNECTED_SPEECH.value in reasons

    def test_flag_off_keeps_old_behavior(self):
        score = compute_phoneme_score(
            _segs(["t", "e", "s", "p", "r", "e", "p"]), self.REF, self.SPANS,
            connected_speech_enabled=False,
        )
        assert [e for e in score.errors if e.severity in ("medium", "high")]
        assert score.overall_accuracy < 1.0


# ── Round 2: biến thể giọng máy-đọc/non-rhotic mở rộng ───────────────────────

class TestNonPrevocalicR:
    """/r/ coda ÂM TIẾT (cuối từ / trước phụ âm) được tha ở accent default;
    /r/ trước nguyên âm vẫn bắt."""

    def test_mid_word_r_before_consonant_deletion_accepted(self):
        # "morning" /m ɔː r n ɪ ŋ/ đọc non-rhotic /m ɔː n ɪ ŋ/ → không lỗi.
        ref = ["m", "ɔː", "r", "n", "ɪ", "ŋ"]
        spans = [WordSpan("morning", 0, 6)]
        score = compute_phoneme_score(
            _segs(["m", "ɔː", "n", "ɪ", "ŋ"]), ref, spans,
            accept_accent_variants=True,
        )
        assert score.overall_accuracy == 1.0
        r_point = score.words[0].phonemes[2]
        assert r_point.status == "ok"
        assert r_point.penalty_reason == PenaltyReason.ACCENT_VARIANT.value

    def test_mid_word_r_still_error_when_flag_off(self):
        ref = ["m", "ɔː", "r", "n", "ɪ", "ŋ"]
        spans = [WordSpan("morning", 0, 6)]
        score = compute_phoneme_score(
            _segs(["m", "ɔː", "n", "ɪ", "ŋ"]), ref, spans,
            accept_accent_variants=False,
        )
        assert score.overall_accuracy < 1.0

    def test_r_before_vowel_deletion_still_error(self):
        # "very" /v e r i/ — /r/ trước nguyên âm KHÔNG droppable, nuốt là lỗi thật.
        ref = ["v", "e", "r", "iː"]
        spans = [WordSpan("very", 0, 4)]
        score = compute_phoneme_score(
            _segs(["v", "e", "iː"]), ref, spans, accept_accent_variants=True,
        )
        assert score.overall_accuracy < 1.0
        assert all(
            p.penalty_reason != PenaltyReason.ACCENT_VARIANT.value
            for p in score.words[0].phonemes
        )

    def test_coda_r_heard_as_w_accepted(self):
        # "our" /aʊ ə r/ nghe thành /ɑː w/ — offglide, không phải lỗi (accent default).
        ref = ["aʊ", "ə", "r"]
        spans = [WordSpan("our", 0, 3)]
        score = compute_phoneme_score(
            _segs(["aʊ", "ə", "w"]), ref, spans, accept_accent_variants=True,
        )
        r_point = score.words[0].phonemes[2]
        assert r_point.status == "ok"
        assert r_point.penalty_reason == PenaltyReason.ACCENT_VARIANT.value

    def test_coda_r_heard_as_l_still_error(self):
        ref = ["aʊ", "ə", "r"]
        spans = [WordSpan("our", 0, 3)]
        score = compute_phoneme_score(
            _segs(["aʊ", "ə", "l"]), ref, spans, accept_accent_variants=True,
        )
        assert score.words[0].phonemes[2].status == "sub"


class TestMachineVoiceVariants:
    """with θ↔ð (cả hai đều chuẩn) + /w/ recognizer-prone (deletion → low)."""

    def test_with_said_as_voiced_matches(self):
        # ref /wɪθ/ đọc /wɪð/ → khớp hoàn toàn (cả hai biến thể đều trong từ điển).
        ref = ["w", "ɪ", "θ"]
        spans = [WordSpan("with", 0, 3)]
        score = compute_phoneme_score(_segs(["w", "ɪ", "ð"]), ref, spans)
        assert score.overall_accuracy == 1.0

    def test_with_reverse_direction_matches(self):
        ref = ["w", "ɪ", "ð"]
        spans = [WordSpan("with", 0, 3)]
        score = compute_phoneme_score(_segs(["w", "ɪ", "θ"]), ref, spans)
        assert score.overall_accuracy == 1.0

    def test_theta_eth_swap_other_words_still_flagged(self):
        # "myth" θ→ð KHÔNG phải biến thể của từ khác → vẫn là lỗi (medium+).
        ref = ["m", "ɪ", "θ"]
        spans = [WordSpan("myth", 0, 3)]
        score = compute_phoneme_score(_segs(["m", "ɪ", "ð"]), ref, spans)
        assert score.overall_accuracy < 1.0

    def test_w_deletion_is_low_severity(self):
        # "website" wav2vec nuốt glide /w/ ngay cả với giọng máy → recognizer-prone,
        # deletion severity low (kể cả ở onset) — không đỏ, vào hidden-noise.
        from src.phoneme.ipa import deletion_severity

        assert deletion_severity("w", is_onset=True) == "low"
        assert deletion_severity("w", is_onset=False) == "low"
        # Các onset consonant thường vẫn high — không nới nhầm cả lớp.
        assert deletion_severity("k", is_onset=True) == "high"


class TestLearnerRegressionGuards:
    """Chốt guard bằng test: lỗi học viên thật KHÔNG bị các fix mới nuốt mất."""

    def test_final_s_drop_still_flagged(self):
        # "nice day" đọc thiếu /s/ — với mọi fix bật (default) vẫn phải ra lỗi.
        ref = ["n", "aɪ", "s", "d", "eɪ"]
        spans = [WordSpan("nice", 0, 3), WordSpan("day", 3, 5)]
        score = compute_phoneme_score(_segs(["n", "aɪ", "d", "eɪ"]), ref, spans)
        assert [e for e in score.errors if e.severity in ("medium", "high")]
        assert score.overall_accuracy < 1.0

    def test_th_stopping_still_flagged(self):
        # th→t (θ→t, lỗi VN kinh điển trong _REAL_ERROR_SUBS) conf cao → vẫn lỗi.
        ref = ["θ", "ɪ", "n"]
        spans = [WordSpan("thin", 0, 3)]
        score = compute_phoneme_score(_segs(["t", "ɪ", "n"]), ref, spans)
        subs = [e for e in score.errors if e.expected == "θ"]
        assert subs and subs[0].severity in ("medium", "high")

    def test_final_stop_drop_before_vowel_still_flagged(self):
        # "test out": nuốt /t/ trước NGUYÊN ÂM (phải nối âm) → vẫn lỗi.
        ref = ["t", "e", "s", "t", "aʊ", "t"]
        spans = [WordSpan("test", 0, 4), WordSpan("out", 4, 6)]
        score = compute_phoneme_score(_segs(["t", "e", "s", "aʊ", "t"]), ref, spans)
        assert [e for e in score.errors if e.severity in ("medium", "high")]
