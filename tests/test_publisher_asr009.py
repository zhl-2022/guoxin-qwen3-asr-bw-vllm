import importlib.util
from pathlib import Path
import sys
import unittest

BASE=Path(__file__).resolve().parents[1]/'scripts/submission009'
sys.path.insert(0,str(BASE))
from evaluate_asr import edit_counts,parse_transcription,score


class ASREvaluationTests(unittest.TestCase):
    def test_empty_hypothesis_counts_all_words(self):
        self.assertEqual(edit_counts('one two',''),dict(substitutions=0,deletions=2,insertions=0,reference_words=2))

    def test_insertions_on_empty_reference(self):
        self.assertEqual(edit_counts('','one two'),dict(substitutions=0,deletions=0,insertions=2,reference_words=0))

    def test_unequal_length_alignment(self):
        result=edit_counts('a b c d','a x c d e')
        self.assertEqual(result,dict(substitutions=1,deletions=0,insertions=1,reference_words=4))

    def test_malformed_protocol_is_rejected(self):
        for value in ('Hello','English<asr_text>hello','language English<asr_text>a<asr_text>b',None):
            with self.assertRaises(ValueError):parse_transcription(value)

    def test_language_and_text_are_preserved(self):
        self.assertEqual(parse_transcription('language English<asr_text>Mr. Jones, 21!'),('English','Mr. Jones, 21!'))

    def fixture(self):
        samples=[dict(id='a',reference='hello world',audio_sha256='hash-a'),dict(id='b',reference='red green blue',audio_sha256='hash-b')]
        predictions=[dict(id='a',prediction='HELLO WORLD.',status='ok',audio_sha256='hash-a'),dict(id='b',prediction='red green black',status='ok',audio_sha256='hash-b')]
        return samples,predictions

    def test_micro_wer_not_mean_utterance_wer(self):
        samples,preds=self.fixture()
        records,result=score(samples,preds)
        self.assertEqual(result['micro_word_error_rate_percent'],20)
        self.assertFalse(result['numeric_comparison_pass'])
        self.assertEqual(records[0]['raw_prediction'],'HELLO WORLD.')

    def test_duplicate_missing_and_extra_ids_rejected(self):
        samples,preds=self.fixture()
        for broken in (preds[:1],preds+[preds[0]],preds+[dict(preds[0],id='extra')]):
            with self.assertRaises(ValueError):score(samples,broken)

    def test_any_failed_request_prevents_pass(self):
        samples,preds=self.fixture()
        preds[1]={'id':'b','status':'failed','error':'timeout'}
        _,result=score(samples,preds)
        self.assertEqual(result['micro_word_error_rate_percent'],0)
        self.assertFalse(result['complete'])
        self.assertFalse(result['numeric_comparison_pass'])

    def test_audio_identity_must_match(self):
        samples,preds=self.fixture()
        preds[0]['audio_sha256']='other'
        with self.assertRaises(ValueError):score(samples,preds)

    def test_relative_threshold_not_two_percentage_points(self):
        samples,preds=self.fixture()
        _,result=score(samples,preds)
        self.assertAlmostEqual(result['limit_percent'],1.6626)
        self.assertFalse(result['publisher_normalization_implementation_published'])
        self.assertFalse(result['official_acceptance'])


if __name__=='__main__':unittest.main()
