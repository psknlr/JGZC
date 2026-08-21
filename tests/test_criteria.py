"""判定引擎：肌少症三标准、骨质疏松诊断与分层、五轴风险。"""

import unittest

from osteosarc_agent.criteria import osteoporosis as op
from osteosarc_agent.criteria import risk
from osteosarc_agent.criteria import sarcopenia as sarco


class SarcopeniaStandardTests(unittest.TestCase):
    """The reference patient: grip 17 kg, 5×STS 16 s, muscle mass not measured."""

    facts = {"sex": "F", "age": 78, "grip_kg": 17, "sts5_s": 16, "gait_speed_ms": 0.72}

    def test_awgs_calls_it_possible_sarcopenia_without_mass(self):
        verdict = sarco.assess("AWGS2019", self.facts)
        self.assertEqual(verdict.verdict, sarco.POSSIBLE)
        self.assertTrue(verdict.actionable)
        self.assertTrue(verdict.strength.is_low)      # 17 < 18
        self.assertTrue(verdict.performance.is_low)   # 16 ≥ 12

    def test_ewgsop2_reaches_probable_through_the_chair_stand(self):
        verdict = sarco.assess("EWGSOP2", self.facts)
        self.assertEqual(verdict.verdict, sarco.PROBABLE)
        # Grip 17 is *not* low under EWGSOP2's 16 kg cutoff; the chair stand is.
        self.assertIn("sts5_s", verdict.strength.measures)
        self.assertTrue(verdict.strength.is_low)

    def test_glis_cannot_conclude_without_muscle_mass(self):
        verdict = sarco.assess("GLIS2024", self.facts)
        self.assertEqual(verdict.verdict, sarco.INDETERMINATE)
        self.assertFalse(verdict.actionable)
        self.assertIn("mass", verdict.missing)

    def test_the_three_standards_disagree_and_the_reason_is_named(self):
        verdicts = sarco.assess_all(self.facts)
        self.assertEqual(len({v.verdict for v in verdicts}), 3)
        notes = " ".join(sarco.cross_standard_notes(verdicts, self.facts))
        self.assertIn("17", notes)
        self.assertIn("18", notes)
        self.assertIn("16", notes)

    def test_grip_between_cutoffs_the_other_way(self):
        """27.5 kg: low for AWGS/GLIS (<28), normal for EWGSOP2 (<27)."""
        facts = {"sex": "M", "age": 72, "grip_kg": 27.5, "asmi": 6.8, "gait_speed_ms": 1.05}
        awgs = sarco.assess("AWGS2019", facts)
        ewgsop2 = sarco.assess("EWGSOP2", facts)
        glis = sarco.assess("GLIS2024", facts)
        self.assertEqual(awgs.verdict, sarco.SARCOPENIA)
        self.assertEqual(ewgsop2.verdict, sarco.NOT_SARCOPENIA)
        self.assertEqual(glis.verdict, sarco.SARCOPENIA)

    def test_confirmed_diagnosis_when_mass_is_low(self):
        facts = {"sex": "F", "age": 78, "grip_kg": 15, "sts5_s": 16, "asmi": 4.8,
                 "gait_speed_ms": 0.6}
        verdict = sarco.assess("AWGS2019", facts)
        self.assertEqual(verdict.verdict, sarco.SEVERE)
        self.assertIn(verdict.verdict, sarco.CONFIRMED)

    def test_normal_mass_rules_out_and_says_to_look_elsewhere(self):
        facts = {"sex": "F", "age": 70, "grip_kg": 15, "asmi": 6.5, "muscle_mass_method": "DXA"}
        verdict = sarco.assess("AWGS2019", facts)
        self.assertEqual(verdict.verdict, sarco.NOT_SARCOPENIA)
        self.assertTrue(any("其他原因" in note for note in verdict.notes))

    def test_no_measurements_is_indeterminate_for_every_standard(self):
        for verdict in sarco.assess_all({"sex": "F", "age": 80}):
            self.assertEqual(verdict.verdict, sarco.INDETERMINATE)

    def test_unknown_sex_cannot_pick_a_cutoff(self):
        verdict = sarco.assess("AWGS2019", {"age": 78, "grip_kg": 17})
        self.assertEqual(verdict.strength.result, "unknown")

    def test_bia_uses_its_own_female_cutoff(self):
        # 5.5 kg/m² is low by the BIA cutoff (5.7) but not by the DXA one (5.4).
        facts = {"sex": "F", "age": 75, "asmi": 5.5, "grip_kg": 15}
        self.assertTrue(sarco.assess("AWGS2019", {**facts, "muscle_mass_method": "BIA"}).mass.is_low)
        self.assertFalse(sarco.assess("AWGS2019", {**facts, "muscle_mass_method": "DXA"}).mass.is_low)

    def test_glis_declares_its_borrowed_cutoffs(self):
        verdict = sarco.assess("GLIS2024", {"sex": "F", "age": 78, "grip_kg": 17, "asmi": 4.9})
        self.assertTrue(any("借用" in note for note in verdict.notes))

    def test_unknown_standard_raises(self):
        with self.assertRaises(ValueError):
            sarco.assess("EWGSOP1", {})


class OsteoporosisTests(unittest.TestCase):
    def test_vertebral_fracture_diagnoses_regardless_of_bmd(self):
        verdict = op.assess({"sex": "F", "age": 78, "fracture_sites": ["椎体"], "lumbar_tscore": -1.4})
        self.assertEqual(verdict.diagnosis, op.OSTEOPOROSIS)
        self.assertIn("脆性骨折", verdict.basis)

    def test_tscore_alone_diagnoses(self):
        verdict = op.assess({"sex": "F", "age": 70, "postmenopausal": True, "lumbar_tscore": -2.7})
        self.assertEqual(verdict.diagnosis, op.OSTEOPOROSIS)

    def test_low_bone_mass_band(self):
        verdict = op.assess({"sex": "F", "age": 70, "lumbar_tscore": -1.8})
        self.assertEqual(verdict.diagnosis, op.LOW_BONE_MASS)

    def test_no_dxa_and_no_fracture_is_indeterminate(self):
        self.assertEqual(op.assess({"sex": "F", "age": 70}).diagnosis, op.INDETERMINATE)

    def test_hip_site_replaces_inflated_lumbar_reading(self):
        verdict = op.assess({"sex": "F", "age": 78, "fracture_sites": ["椎体"],
                             "lumbar_tscore": -2.9, "femoral_neck_tscore": -2.6})
        self.assertEqual(verdict.site_used, "股骨颈")
        self.assertEqual(verdict.tscore_used, -2.6)
        self.assertTrue(any("虚高" in c for c in verdict.caveats))

    def test_lowest_site_used_when_spine_is_reliable(self):
        verdict = op.assess({"sex": "F", "age": 68, "lumbar_tscore": -2.9, "femoral_neck_tscore": -2.1})
        self.assertEqual(verdict.tscore_used, -2.9)

    def test_supportive_site_fracture_needs_low_bone_mass(self):
        verdict = op.assess({"sex": "F", "age": 70, "fracture_sites": ["前臂"], "lumbar_tscore": -1.6})
        self.assertEqual(verdict.diagnosis, op.OSTEOPOROSIS)
        verdict2 = op.assess({"sex": "F", "age": 70, "fracture_sites": ["前臂"], "lumbar_tscore": -0.4})
        self.assertEqual(verdict2.diagnosis, op.NORMAL)

    def test_very_high_risk_drivers_are_named(self):
        verdict = op.assess({"sex": "F", "age": 78, "fracture_sites": ["椎体"],
                             "lumbar_tscore": -2.9, "femoral_neck_tscore": -2.6, "falls_12m": 2})
        self.assertTrue(verdict.very_high_risk)
        labels = [driver.label for driver in verdict.risk_drivers]
        self.assertIn("既往髋部或椎体脆性骨折", labels)

    def test_severe_requires_both_fracture_and_tscore(self):
        verdict = op.assess({"sex": "F", "age": 78, "fracture_sites": ["椎体"], "femoral_neck_tscore": -2.6})
        self.assertTrue(verdict.severe)
        verdict2 = op.assess({"sex": "F", "age": 78, "fracture_sites": ["椎体"], "femoral_neck_tscore": -1.9})
        self.assertFalse(verdict2.severe)

    def test_height_loss_without_imaging_is_flagged(self):
        verdict = op.assess({"sex": "F", "age": 78, "height_loss_cm": 5, "lumbar_tscore": -2.0})
        self.assertTrue(any("VFA" in c for c in verdict.caveats))


class RiskTests(unittest.TestCase):
    def test_five_axes_are_produced(self):
        axes = risk.profile({"age": 78, "sex": "F"})
        self.assertEqual([a.axis for a in axes],
                         ["fracture", "fall", "sarcopenia", "nutrition", "functional"])

    def test_drivers_carry_their_weights_and_levers(self):
        axes = {a.axis: a for a in risk.profile({"age": 78, "sex": "F", "falls_12m": 2, "frid_count": 2})}
        fall = axes["fall"]
        self.assertTrue(fall.drivers)
        self.assertTrue(all("points" in d for d in fall.drivers))
        self.assertTrue(any("用药重整" in lever for lever in fall.levers))

    def test_unmeasured_axis_is_not_reported_as_low(self):
        axes = {a.axis: a for a in risk.profile({"age": 82, "sex": "M"})}
        self.assertEqual(axes["nutrition"].tier, "unassessed")
        self.assertIn("没测", axes["nutrition"].note)

    def test_measured_and_genuinely_low_stays_low(self):
        facts = {"age": 66, "sex": "F", "mna_sf": 14, "weight_loss_3m_pct": 0,
                 "protein_g_per_kg": 1.3, "bmi": 23, "albumin": 42}
        axes = {a.axis: a for a in risk.profile(facts)}
        self.assertEqual(axes["nutrition"].tier, "low")
        self.assertEqual(axes["nutrition"].confidence, 1.0)

    def test_severe_patient_scores_high_without_pegging_every_axis(self):
        # Includes the flags the diagnosis agent writes back, because that is
        # the state the risk model actually runs against in the pipeline.
        facts = {"age": 78, "sex": "F", "tscore_min": -2.9, "prior_fragility_fracture": True,
                 "falls_12m": 2, "fall_with_injury": True, "gait_speed_ms": 0.72,
                 "sts5_s": 16, "grip_kg": 17, "bmi": 19.9, "mna_sf": 9, "appetite_loss": True,
                 "protein_g_per_kg": 0.8, "frid_count": 1, "dx_osteoporosis": True,
                 "dx_possible_sarcopenia": True, "low_muscle_strength": True,
                 "low_physical_performance": True, "high_fall_risk": True,
                 "vitd_deficient": True, "resistance_training": False,
                 "iadl_dependent": True, "sedentary_hours": 9, "comorbidity_count": 4,
                 "fear_of_falling": True, "dizziness": True, "vision_impairment": True,
                 "uses_walking_aid": True, "living_alone": True, "calcium_intake_mg": 520,
                 "dairy_servings": 0.5, "albumin": 34, "sun_exposure_low": True,
                 "weight_loss_3m_pct": 4, "calf_circumference_cm": 29, "sarc_f": 5, "egfr": 32}
        scores = [a.score for a in risk.profile(facts)]
        self.assertTrue(all(70 <= s <= 100 for s in scores), scores)
        # Nothing pegs at the ceiling — the caps leave headroom above a severe
        # but realistic patient, so the radar still has a shape.
        self.assertTrue(all(s < 100 for s in scores), scores)

    def test_confidence_reflects_missing_inputs(self):
        axes = {a.axis: a for a in risk.profile({"age": 70, "sex": "F"})}
        self.assertLess(axes["sarcopenia"].confidence, 0.5)
        self.assertTrue(axes["sarcopenia"].missing)

    def test_radar_payload_carries_the_model_caveat(self):
        payload = risk.as_radar(risk.profile({"age": 70, "sex": "F"}))
        self.assertIn("不是经验证的预测模型", payload["model_note"])
        self.assertEqual(len(payload["axes"]), 5)


if __name__ == "__main__":
    unittest.main()
