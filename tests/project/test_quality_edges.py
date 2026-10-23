import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from evaluation.solution import score, quality_gate
from src.promote import promote
from test_eval import rows, outputs, report


class ScoringEdges(unittest.TestCase):
    def test_duplicate_json_key_rejected_even_when_values_equal(self):
        out=outputs(rows(1));out[0]['raw']='{"answer":"B","answer":"B"}'
        self.assertEqual(score(rows(1),out)['overall']['valid'],0)

    def test_generation_error_overrules_well_formed_answer(self):
        out=outputs(rows(1));out[0]['error']='timeout after partial output'
        self.assertEqual(score(rows(1),out)['overall'],dict(n=1,correct=0,valid=0,errors=1,accuracy=0.0,format_rate=0.0))

    def test_json_constants_and_nonstring_output_do_not_vanish(self):
        for raw in ['{"answer":NaN}', '{"answer":"B","x":Infinity}', None, ['B']]:
            with self.subTest(raw=raw):
                result=score(rows(1),[dict(id='q0',raw=raw,error=None)])
                self.assertEqual((result['overall']['n'],result['overall']['valid']), (1,0))

    def test_slice_membership_cannot_change_under_equal_counts(self):
        a,b=report(),report();b['per_item'][0]['slices']=['new-group']
        with self.assertRaises(ValueError):quality_gate(a,b,['s0'])

    def test_raw_and_parsed_disagreement_rejected(self):
        b=report();b['per_item'][0]['raw']='{"answer":"A"}'
        with self.assertRaises(ValueError):quality_gate(report(),b,['s0'])

    def test_real_critical_regression_with_overall_improvement(self):
        a=report(20); b=report()
        out=outputs(rows(),0)
        for i in list(range(1,20))+[41,44,47]:out[i]['raw']='{"answer":"B"}'
        b.update(score(rows(),out))
        self.assertGreater(b['overall']['accuracy'],a['overall']['accuracy'])
        self.assertLess(b['slices']['s0']['accuracy'],a['slices']['s0']['accuracy'])
        with self.assertRaises(ValueError):quality_gate(a,b,['s0'])

    def test_bad_revision_type_is_value_error(self):
        b=report();b['model']={'kind':'base','name':'x','revision':None}
        with self.assertRaises(ValueError):quality_gate(report(),b,['s0'])


class RegistrationGuard(unittest.TestCase):
    def test_rejected_loss_never_calls_registry(self):
        client=Mock()
        for final,base,threshold,status in [(2.,1.,1.,'FINISHED'),(float('nan'),1.,1.,'FINISHED'),(-1.,1.,1.,'FINISHED'),(.5,1.,1.,'FAILED')]:
            run=SimpleNamespace(data=SimpleNamespace(metrics={'final_val_loss':final,'base_val_loss':base}),info=SimpleNamespace(status=status,run_id='test'))
            with self.subTest(final=final,status=status), patch('src.promote.mlflow.register_model') as register:
                with self.assertRaises(ValueError):promote(client,run,'model','candidate',threshold)
                register.assert_not_called()
        client.set_registered_model_alias.assert_not_called()
