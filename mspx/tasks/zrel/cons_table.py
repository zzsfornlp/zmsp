#

# constraint tables

# some helpers
e_ent = ['ORG', 'PER', 'GPE']  # entity
e_place = ['GPE', 'LOC', 'FAC']  # place

FULL_TYPES = {
'evt': ['Conflict:Attack', 'Movement:Transport', 'Life:Die', 'Contact:Meet', 'Personnel:End-Position', 'Personnel:Elect', 'Life:Injure', 'Transaction:Transfer-Money', 'Contact:Phone-Write', 'Justice:Trial-Hearing', 'Personnel:Start-Position', 'Justice:Charge-Indict', 'Transaction:Transfer-Ownership', 'Justice:Sentence', 'Justice:Arrest-Jail', 'Life:Marry', 'Justice:Sue', 'Justice:Convict', 'Conflict:Demonstrate', 'Justice:Release-Parole', 'Life:Be-Born', 'Business:Declare-Bankruptcy', 'Justice:Appeal', 'Business:End-Org', 'Business:Start-Org', 'Justice:Fine', 'Life:Divorce', 'Business:Merge-Org', 'Justice:Execute', 'Personnel:Nominate', 'Justice:Extradite', 'Justice:Acquit', 'Justice:Pardon'],
'ef': ['PER', 'GPE', 'ORG', 'FAC', 'LOC', 'VEH', 'WEA'],
'rel': ['ORG-AFF', 'PHYS', 'PART-WHOLE', 'PER-SOC', 'GEN-AFF', 'ART'],
}

CONS_TABLE = {
'evt': [
    ('Business:Declare-Bankruptcy', 'Org', e_ent),
    ('Business:Declare-Bankruptcy', 'Place', e_place),
    ('Business:End-Org', 'Org', 'ORG'),
    ('Business:End-Org', 'Place', e_place),
    ('Business:Merge-Org', 'Org', 'ORG'),
    ('Business:Merge-Org', 'Place', e_place),
    ('Business:Start-Org', 'Agent', e_ent),
    ('Business:Start-Org', 'Org', 'ORG'),
    ('Business:Start-Org', 'Place', e_place),
    ('Conflict:Attack', 'Attacker', e_ent),
    ('Conflict:Attack', 'Instrument', ['WEA', 'VEH']),
    ('Conflict:Attack', 'Place', e_place),
    ('Conflict:Attack', 'Target', ['PER', 'ORG', 'VEH', 'FAC', 'WEA']),
    ('Conflict:Demonstrate', 'Entity', ['PER', 'ORG']),
    ('Conflict:Demonstrate', 'Place', e_place),
    ('Contact:Meet', 'Entity', e_ent),
    ('Contact:Meet', 'Place', e_place),
    ('Contact:Phone-Write', 'Entity', e_ent),
    ('Contact:Phone-Write', 'Place', e_place),
    ('Justice:Acquit', 'Adjudicator', e_ent),
    ('Justice:Acquit', 'Defendant', e_ent),
    ('Justice:Acquit', 'Place', e_place),
    ('Justice:Appeal', 'Adjudicator', e_ent),
    ('Justice:Appeal', 'Defendant', e_ent),
    ('Justice:Appeal', 'Place', e_place),
    ('Justice:Appeal', 'Plaintiff', e_ent),  # note: not in guidelines but annotated!
    ('Justice:Appeal', 'Prosecutor', e_ent),
    ('Justice:Arrest-Jail', 'Agent', e_ent),
    ('Justice:Arrest-Jail', 'Person', 'PER'),
    ('Justice:Arrest-Jail', 'Place', e_place),
    ('Justice:Charge-Indict', 'Adjudicator', e_ent),
    ('Justice:Charge-Indict', 'Defendant', e_ent),
    ('Justice:Charge-Indict', 'Place', e_place),
    ('Justice:Charge-Indict', 'Prosecutor', e_ent),
    ('Justice:Convict', 'Adjudicator', e_ent),
    ('Justice:Convict', 'Defendant', e_ent),
    ('Justice:Convict', 'Place', e_place),
    ('Justice:Execute', 'Agent', e_ent),
    ('Justice:Execute', 'Person', 'PER'),
    ('Justice:Execute', 'Place', e_place),
    ('Justice:Extradite', 'Agent', e_ent),
    ('Justice:Extradite', 'Destination', e_place),
    ('Justice:Extradite', 'Origin', e_place),
    ('Justice:Extradite', 'Person', 'PER'),
    ('Justice:Fine', 'Adjudicator', e_ent),
    ('Justice:Fine', 'Entity', e_ent),
    ('Justice:Fine', 'Place', e_place),
    ('Justice:Pardon', 'Adjudicator', e_ent),
    ('Justice:Pardon', 'Defendant', e_ent),
    ('Justice:Pardon', 'Place', e_place),
    ('Justice:Release-Parole', 'Entity', e_ent),
    ('Justice:Release-Parole', 'Person', 'PER'),
    ('Justice:Release-Parole', 'Place', e_place),
    ('Justice:Sentence', 'Adjudicator', e_ent),
    ('Justice:Sentence', 'Defendant', e_ent),
    ('Justice:Sentence', 'Place', e_place),
    ('Justice:Sue', 'Adjudicator', e_ent),
    ('Justice:Sue', 'Defendant', e_ent),
    ('Justice:Sue', 'Place', e_place),
    ('Justice:Sue', 'Plaintiff', e_ent),
    ('Justice:Trial-Hearing', 'Adjudicator', e_ent),
    ('Justice:Trial-Hearing', 'Defendant', e_ent),
    ('Justice:Trial-Hearing', 'Place', e_place),
    ('Justice:Trial-Hearing', 'Prosecutor', e_ent),
    ('Life:Be-Born', 'Person', 'PER'),
    ('Life:Be-Born', 'Place', e_place),
    ('Life:Die', 'Agent', e_ent),
    ('Life:Die', 'Instrument', ['WEA', 'VEH']),
    ('Life:Die', 'Place', e_place),
    ('Life:Die', 'Victim', 'PER'),
    ('Life:Divorce', 'Person', 'PER'),
    ('Life:Divorce', 'Place', e_place),
    ('Life:Injure', 'Agent', e_ent),
    ('Life:Injure', 'Instrument', ['WEA', 'VEH']),
    ('Life:Injure', 'Place', e_place),
    ('Life:Injure', 'Victim', 'PER'),
    ('Life:Marry', 'Person', 'PER'),
    ('Life:Marry', 'Place', e_place),
    ('Movement:Transport', 'Agent', e_ent),
    ('Movement:Transport', 'Artifact', ['PER', 'WEA', 'VEH']),
    ('Movement:Transport', 'Destination', e_place),
    ('Movement:Transport', 'Origin', e_place),
    ('Movement:Transport', 'Vehicle', 'VEH'),
    ('Personnel:Elect', 'Entity', e_ent),
    ('Personnel:Elect', 'Person', 'PER'),
    ('Personnel:Elect', 'Place', e_place),
    ('Personnel:End-Position', 'Entity', ['ORG', 'GPE']),
    ('Personnel:End-Position', 'Person', 'PER'),
    ('Personnel:End-Position', 'Place', e_place),
    ('Personnel:Nominate', 'Agent', e_ent + ['FAC']),
    ('Personnel:Nominate', 'Person', 'PER'),
    ('Personnel:Nominate', 'Place', e_place),
    ('Personnel:Start-Position', 'Entity', ['ORG', 'GPE']),
    ('Personnel:Start-Position', 'Person', 'PER'),
    ('Personnel:Start-Position', 'Place', e_place),
    ('Transaction:Transfer-Money', 'Beneficiary', e_ent),
    ('Transaction:Transfer-Money', 'Giver', e_ent),
    ('Transaction:Transfer-Money', 'Place', e_place),
    ('Transaction:Transfer-Money', 'Recipient', e_ent),
    ('Transaction:Transfer-Ownership', 'Artifact', ['WEA', 'VEH', 'FAC', 'ORG']),
    ('Transaction:Transfer-Ownership', 'Beneficiary', e_ent),
    ('Transaction:Transfer-Ownership', 'Buyer', e_ent),
    ('Transaction:Transfer-Ownership', 'Place', e_place),
    ('Transaction:Transfer-Ownership', 'Seller', e_ent),
],
'rel': [
    (['PER', 'ORG', 'GPE'], 'ORG-AFF', ['ORG', 'GPE']),
    (['PER', 'FAC', 'LOC', 'GPE'], 'PHYS', ['FAC', 'LOC', 'GPE']),
    (['FAC', 'LOC', 'GPE'], 'PART-WHOLE', ['FAC', 'LOC', 'GPE']),
    (['ORG'], 'PART-WHOLE', ['ORG', 'GPE']),
    (['VEH'], 'PART-WHOLE', ['VEH']),
    (['WEA'], 'PART-WHOLE', ['WEA']),
    (['PER'], 'PER-SOC', ['PER']),
    (['PER'], 'GEN-AFF', ['PER', 'LOC', 'GPE', 'ORG']),
    (['ORG'], 'GEN-AFF', ['LOC', 'GPE']),
    (['PER', 'ORG', 'GPE'], 'ART', ['WEA', 'VEH', 'FAC']),
],
}

# full allowed set
def compile_full_set(head_types, tail_types, t_cons):
    ret = set()
    for tH, rel, tT in t_cons:
        tH = head_types if tH is None else tH
        tT = tail_types if tT is None else tT
        if isinstance(tH, str):
            tH = [tH]
        if isinstance(tT, str):
            tT = [tT]
        for a in tH:
            for b in tT:
                ret.add((a, rel, b))
    return ret

MAT_RELS = """
Form_Of		Arg1:Descriptor, 		Arg2:Material|Participating_Material|Descriptor|MStructure|Microstructure|Phase|Operation|Result|Synthesis|Environment|Phenomenon|Characterization|Property
Form_Of		Arg1:MStructure, 		Arg2:Material|Phase|Property|Participating_Material
Form_Of		Arg1:Microstructure, 	Arg2:Material|Phase
Form_Of		Arg1:Phase, 			Arg2:Material|MStructure|Microstructure
Form_Of		Arg1:Synthesis,		Arg2:Synthesis
Form_Of		Arg1:Characterization,	Arg2:Characterization
Form_Of		Arg1:Property,		Arg2:Property

Condition_Of		Arg1:Participating_Material, 	Arg2:Property
Condition_Of		Arg1:Descriptor, 		Arg2:Result|Phase|Property|Characterization
Condition_Of		Arg1:MStructure, 	Arg2:MStructure|Characterization|Phenomenon|Number|Property|Descriptor
Condition_Of		Arg1:Microstructure, 	Arg2:Phenomenon|Property
Condition_Of		Arg1:Phase, 			Arg2:Phase|Operation|Result
Condition_Of		Arg1:Operation, 		Arg2:Material|Operation|Synthesis|Environment|Phenomenon|Characterization|Result
Condition_Of		Arg1:Result, 			Arg2:Material|Descriptor|MStructure|Microstructure|Operation|Result|Environment|Phenomenon|Characterization|Property|Phase|Synthesis
Condition_Of		Arg1:Environment, 		Arg2:Material|Participating_Material|Descriptor|MStructure|Microstructure|Phase|Operation|Result|Synthesis|Environment|Phenomenon|Characterization|Application
Condition_Of		Arg1:Phenomenon, 		Arg2:Material|Phase|Operation|Synthesis|Environment|Phenomenon|Characterization|Descriptor|Result
Condition_Of		Arg1:Property, 		Arg2:Phenomenon|Operation|Property|Result
Condition_Of		Arg1:Time, 		Arg2:Property

Observed_In		Arg1:Participating_Material, 		Arg2:MStructure|Environment
Observed_In		Arg1:Descriptor, 		Arg2:Microstructure
Observed_In		Arg1:MStructure, 		Arg2:Material|Descriptor|MStructure|Operation|Synthesis|Environment|Phenomenon|Characterization|Application
Observed_In		Arg1:Microstructure, 	Arg2:Material|Descriptor|MStructure|Synthesis|Environment|Phenomenon|Characterization|Application
Observed_In		Arg1:Result,			Arg2:Material|Application|MStructure
Observed_In		Arg1:Phase, 			Arg2:Microstructure|MStructure|Material|Participating_Material
Observed_In		Arg1:Environment, 		Arg2:Material|Descriptor|MStructure|Microstructure|Operation|Synthesis|Phenomenon|Characterization|Application
Observed_In		Arg1:Phenomenon, 		Arg2:Material|Participating_Material|Descriptor|MStructure|Microstructure|Operation|Synthesis|Environment|Phenomenon|Characterization|Application
Observed_In		Arg1:Property, 		Arg2:MStructure|Microstructure|Environment|Phenomenon|Characterization|Application
Observed_In		Arg1:Operation, 		Arg2:Synthesis|Characterization|MStructure

Property_Of		Arg1:Property, 		Arg2:Material|Participating_Material|Descriptor|MStructure|Microstructure|Phase|Operation|Phenomenon|Application|Characterization

Input		Arg1:Material, 			Arg2:Operation|Characterization|Application|Synthesis
Input		Arg1:Participating_Material, 	Arg2:Material|Operation|Result|Synthesis|Phase|Phenomenon
Input		Arg1:Descriptor, 			Arg2:Operation|Synthesis|Number
Input		Arg1:MStructure, 			Arg2:Phase|Operation|Result|Phenomenon|Characterization
Input		Arg1:Phase, 			Arg2:Operation|Result|Phenomenon|Characterization
Input		Arg1:Microstructure, 		Arg2:Operation|Result|Phenomenon
Input		Arg1:Operation, 			Arg2:Operation|Synthesis|Phenomenon|Characterization
Input		Arg1:Result, 				Arg2:Operation|Synthesis|Characterization|Phenomenon|Descriptor
Input		Arg1:Synthesis, 			Arg2:Operation|Result
Input		Arg1:Environment, 			Arg2:Operation|Result|Characterization|Property|Phenomenon|MStructure
Input		Arg1:Phenomenon, 			Arg2:Operation|Result|Characterization
Input		Arg1:Characterization, 		Arg2:Operation|Result|Phenomenon|Characterization
Input		Arg1:Property, 			Arg2:Operation|Result|Characterization|Phenomenon

Output		Arg1:Operation, 			Arg2:Material|Participating_Material|Descriptor|MStructure|Microstructure|Phase|Phenomenon|Property|Application
Output		Arg1:Result, 			Arg2:Material|Property|Microstructure|Phenomenon|Phase|Application|Descriptor
Output		Arg1:Synthesis, 			Arg2:Material|Participating_Material|Descriptor|MStructure|Phase|Result|Application
Output		Arg1:Phenomenon, 			Arg2:Material|Descriptor|MStructure|Microstructure|Property|Phase|Environment|Result
Output		Arg1:Characterization, 		Arg2:Material|Descriptor|MStructure|Microstructure|Phase|Operation|Environment|Phenomenon|Property|Application
Output		Arg1:Property, 			Arg2:MStructure

Result_of		Arg1:Phenomenon, 		Arg2:Phenomenon
Result_of		Arg1:Result, 		Arg2:MStructure|Phase|Operation|Synthesis|Environment|Characterization|Property|Phenomenon

Next_Opr		Arg1:Operation, 			Arg2:Operation|Characterization|Synthesis
Next_Opr		Arg1:Characterization, 		Arg2:Operation|Characterization|Synthesis
Next_Opr		Arg1:Synthesis, 			Arg2:Operation|Characterization|Synthesis

Coref		Arg1:Material, 			Arg2:Material
Coref		Arg1:Participating_Material, 	Arg2:Participating_Material
Coref		Arg1:Descriptor, 			Arg2:Descriptor
Coref		Arg1:MStructure, 			Arg2:MStructure
Coref		Arg1:Microstructure, 		Arg2:Microstructure
Coref		Arg1:Phase,	 			Arg2:Phase
Coref		Arg1:Operation, 			Arg2:Operation
Coref		Arg1:Synthesis, 			Arg2:Synthesis
Coref		Arg1:Phenomenon, 			Arg2:Phenomenon
Coref		Arg1:Property, 			Arg2:Property
Coref		Arg1:Characterization, 		Arg2:Characterization
Coref		Arg1:Application, 			Arg2:Application
Coref		Arg1:Environment, 			Arg2:Environment

Number_of	Arg1:Number, 			Arg2:Amount_Unit|Participating_Material|MStructure|Result|Material|Property|Environment|Synthesis

Amount_of	Arg1:Amount_Unit, 			Arg2:Participating_Material|Phase|MStructure|Operation|Result|Synthesis|Environment|Characterization|Property|Application
"""

def compile_mat_set():
    ret = set()
    for line in MAT_RELS.split('\n'):
        line = line.strip()
        if line:
            r, hs, ts = line.split()
            hs = hs.split(":")[-1].split(",")[0].split("|")
            ts = ts.split(":")[-1].split("|")
            for a in hs:
                for b in ts:
                    ret.add((a, r, b))
    return ret

# --
CONS_SET = {
    'evt': compile_full_set(FULL_TYPES['evt'], FULL_TYPES['ef'], CONS_TABLE['evt']),
    'rel': compile_full_set(FULL_TYPES['ef'], FULL_TYPES['ef'], CONS_TABLE['rel']),
    'mat': compile_mat_set(),
}
# evt: train=13/6189=0.002, dev=0/418=0, test=4/806=0.005
# rel: train=42/6695=0.006, dev=2/459=0.004, test=10/749=0.013
# mat: v221222=11/3885=0.003
# --
