#

# label budgets of various
# note: for ace/ere, only allow contact.entity and marry/divorce.person to be 2!

ACE_ARG_BUDGETS = {
'Business.Declare-Bankruptcy': {'Org': 1, 'Time-After': 1, 'Place': 1, 'Time-Within': 1, 'Time-At-Beginning': 1},
'Business.End-Org': {'Org': 1, 'Time-Holds': 1, 'Place': 1, 'Time-Within': 1, 'Time-At-Beginning': 1, 'Time-After': 1},
'Business.Merge-Org': {'Org': 1, 'Time-Ending': 1},
'Business.Start-Org': {'Org': 1, 'Agent': 1, 'Place': 1, 'Time-Within': 1, 'Time-After': 1, 'Time-Starting': 1, 'Time-Before': 1},
'Conflict.Attack': {'Attacker': 1, 'Place': 1, 'Target': 1, 'Time-Within': 1, 'Instrument': 1, 'Time-Ending': 1, 'Time-Holds': 1, 'Time-At-Beginning': 1, 'Victim': 1, 'Time-After': 1, 'Time-Starting': 1, 'Time-Before': 1, 'Agent': 1, 'Time-At-End': 1},
'Conflict.Demonstrate': {'Place': 1, 'Entity': 1, 'Time-Within': 1, 'Time-At-End': 1, 'Time-Starting': 1},
'Contact.Meet': {'Entity': 2, 'Place': 1, 'Time-Within': 1, 'Time-Starting': 1, 'Time-Holds': 1, 'Time-At-Beginning': 1, 'Time-Ending': 1, 'Time-After': 1},
'Contact.Phone-Write': {'Entity': 2, 'Time-Within': 1, 'Time-Before': 1, 'Place': 1, 'Time-Holds': 1, 'Time-Starting': 1, 'Time-After': 1},
'Justice.Acquit': {'Defendant': 1, 'Adjudicator': 1, 'Time-Within': 1, 'Crime': 1},
'Justice.Appeal': {'Place': 1, 'Adjudicator': 1, 'Plaintiff': 1, 'Time-Within': 1, 'Crime': 1, 'Time-Holds': 1},
'Justice.Arrest-Jail': {'Person': 1, 'Agent': 1, 'Time-Within': 1, 'Place': 1, 'Crime': 1, 'Time-Holds': 1, 'Time-Ending': 1, 'Time-At-Beginning': 1, 'Time-Starting': 1, 'Time-Before': 1},
'Justice.Charge-Indict': {'Adjudicator': 1, 'Defendant': 1, 'Crime': 1, 'Prosecutor': 1, 'Time-Within': 1, 'Place': 1, 'Time-Ending': 1, 'Time-Before': 1},
'Justice.Convict': {'Defendant': 1, 'Crime': 1, 'Adjudicator': 1, 'Time-Within': 1, 'Place': 1, 'Time-At-Beginning': 1},
'Justice.Execute': {'Agent': 1, 'Place': 1, 'Person': 1, 'Time-After': 1, 'Time-Within': 1, 'Crime': 1, 'Time-At-Beginning': 1},
'Justice.Extradite': {'Destination': 1, 'Origin': 1, 'Person': 1, 'Agent': 1, 'Time-Within': 1},
'Justice.Fine': {'Entity': 1, 'Money': 1, 'Place': 1, 'Time-Within': 1, 'Crime': 1, 'Adjudicator': 1},
'Justice.Pardon': {'Defendant': 1, 'Adjudicator': 1, 'Place': 1, 'Time-At-End': 1},
'Justice.Release-Parole': {'Person': 1, 'Entity': 1, 'Time-Within': 1, 'Place': 1, 'Crime': 1, 'Time-After': 1},
'Justice.Sentence': {'Adjudicator': 1, 'Defendant': 1, 'Sentence': 1, 'Crime': 1, 'Place': 1, 'Time-Within': 1, 'Time-At-End': 1, 'Time-Starting': 1},
'Justice.Sue': {'Plaintiff': 1, 'Defendant': 1, 'Place': 1, 'Time-Within': 1, 'Crime': 1, 'Adjudicator': 1, 'Time-Holds': 1},
'Justice.Trial-Hearing': {'Prosecutor': 1, 'Place': 1, 'Time-Within': 1, 'Defendant': 1, 'Crime': 1, 'Adjudicator': 1, 'Time-Starting': 1, 'Time-At-End': 1, 'Time-Holds': 1},
'Life.Be-Born': {'Place': 1, 'Person': 1, 'Time-Within': 1, 'Time-Holds': 1},
'Life.Die': {'Victim': 1, 'Agent': 1, 'Place': 1, 'Time-Within': 1, 'Instrument': 1, 'Time-Before': 1, 'Time-After': 1, 'Person': 1, 'Time-Starting': 1, 'Time-Holds': 1, 'Time-Ending': 1, 'Time-At-Beginning': 1},
'Life.Divorce': {'Person': 2, 'Time-Within': 1, 'Place': 1},
'Life.Injure': {'Agent': 1, 'Victim': 1, 'Place': 1, 'Time-Within': 1, 'Instrument': 1},
'Life.Marry': {'Person': 2, 'Time-Within': 1, 'Place': 1, 'Time-Holds': 1, 'Time-Before': 1},
'Movement.Transport': {'Vehicle': 1, 'Artifact': 1, 'Destination': 1, 'Agent': 1, 'Origin': 1, 'Time-At-Beginning': 1, 'Time-Within': 1, 'Time-Holds': 1, 'Time-At-End': 1, 'Time-Starting': 1, 'Time-Ending': 1, 'Time-After': 1, 'Victim': 1, 'Place': 1, 'Time-Before': 1},
'Personnel.Elect': {'Person': 1, 'Position': 1, 'Entity': 1, 'Place': 1, 'Time-Within': 1, 'Time-Starting': 1, 'Time-Holds': 1, 'Time-Before': 1, 'Time-At-Beginning': 1},
'Personnel.End-Position': {'Person': 1, 'Entity': 1, 'Position': 1, 'Time-Within': 1, 'Place': 1, 'Time-Ending': 1, 'Time-Holds': 1, 'Time-At-End': 1, 'Time-Before': 1, 'Time-Starting': 1, 'Time-After': 1},
'Personnel.Nominate': {'Person': 1, 'Agent': 1, 'Time-Within': 1, 'Position': 1},
'Personnel.Start-Position': {'Person': 1, 'Entity': 1, 'Position': 1, 'Place': 1, 'Time-Within': 1, 'Time-At-Beginning': 1, 'Time-After': 1, 'Time-Before': 1, 'Time-Starting': 1, 'Time-Holds': 1},
'Transaction.Transfer-Money': {'Giver': 1, 'Recipient': 1, 'Money': 1, 'Beneficiary': 1, 'Time-Holds': 1, 'Time-Within': 1, 'Time-After': 1, 'Time-Starting': 1, 'Place': 1, 'Time-Before': 1},
'Transaction.Transfer-Ownership': {'Artifact': 1, 'Buyer': 1, 'Beneficiary': 1, 'Seller': 1, 'Time-Within': 1, 'Place': 1, 'Time-Before': 1, 'Price': 1, 'Time-Ending': 1, 'Time-At-Beginning': 1},
}
# --
for k in ACE_ARG_BUDGETS:
    ACE_ARG_BUDGETS[k]["Time"] = 1  # extra shorter Time arg!
# --

ERE_ARG_BUDGETS = {
'business.declarebankruptcy': {'org': 1, 'time': 1},
'business.endorg': {'org': 1, 'place': 1, 'time': 1},
'business.mergeorg': {'org': 1},
'business.startorg': {'org': 1, 'agent': 1, 'place': 1, 'time': 1},
'conflict.attack': {'attacker': 1, 'target': 1, 'place': 1, 'instrument': 1, 'time': 1},
'conflict.demonstrate': {'entity': 1, 'time': 1, 'place': 1},
'contact.broadcast': {'entity': 1, 'time': 1, 'place': 1, 'audience': 1},
'contact.contact': {'entity': 2, 'time': 1, 'place': 1},
'contact.correspondence': {'entity': 2, 'time': 1, 'place': 1},
'contact.meet': {'entity': 2, 'place': 1, 'time': 1},
'justice.acquit': {'defendant': 1, 'adjudicator': 1, 'time': 1, 'crime': 1, 'place': 1},
'justice.appeal': {'defendant': 1, 'crime': 1, 'adjudicator': 1, 'prosecutor': 1, 'place': 1},
'justice.arrestjail': {'person': 1, 'place': 1, 'time': 1, 'agent': 1, 'crime': 1},
'justice.chargeindict': {'defendant': 1, 'crime': 1, 'time': 1, 'adjudicator': 1, 'prosecutor': 1, 'place': 1},
'justice.convict': {'defendant': 1, 'adjudicator': 1, 'crime': 1, 'time': 1, 'place': 1},
'justice.execute': {'person': 1, 'crime': 1, 'place': 1, 'agent': 1, 'time': 1},
'justice.extradite': {'destination': 1, 'person': 1, 'agent': 1, 'origin': 1, 'crime': 1, 'time': 1},
'justice.fine': {'entity': 1, 'money': 1, 'crime': 1, 'place': 1, 'adjudicator': 1, 'time': 1},
'justice.pardon': {'defendant': 1, 'adjudicator': 1, 'crime': 1, 'place': 1, 'time': 1},
'justice.releaseparole': {'person': 1, 'place': 1, 'time': 1, 'agent': 1, 'crime': 1},
'justice.sentence': {'defendant': 1, 'crime': 1, 'sentence': 1, 'time': 1, 'place': 1, 'adjudicator': 1},
'justice.sue': {'plaintiff': 1, 'defendant': 1, 'place': 1, 'time': 1, 'crime': 1, 'adjudicator': 1},
'justice.trialhearing': {'defendant': 1, 'time': 1, 'crime': 1, 'place': 1, 'prosecutor': 1, 'adjudicator': 1},
'life.beborn': {'person': 1, 'place': 1, 'time': 1},
'life.die': {'agent': 1, 'victim': 1, 'place': 1, 'time': 1, 'instrument': 1},
'life.divorce': {'person': 2, 'time': 1, 'place': 1},
'life.injure': {'agent': 1, 'victim': 1, 'instrument': 1, 'place': 1, 'time': 1},
'life.marry': {'person': 2, 'time': 1, 'place': 1},
'manufacture.artifact': {'agent': 1, 'artifact': 1, 'place': 1, 'instrument': 1, 'time': 1},
'movement.transportartifact': {'agent': 1, 'artifact': 1, 'time': 1, 'destination': 1, 'origin': 1, 'instrument': 1},
'movement.transportperson': {'person': 1, 'destination': 1, 'time': 1, 'agent': 1, 'origin': 1, 'instrument': 1},
'personnel.elect': {'person': 1, 'agent': 1, 'place': 1, 'time': 1, 'position': 1},
'personnel.endposition': {'person': 1, 'entity': 1, 'position': 1, 'place': 1, 'time': 1},
'personnel.nominate': {'person': 1, 'position': 1, 'agent': 1, 'time': 1},
'personnel.startposition': {'person': 1, 'entity': 1, 'position': 1, 'time': 1, 'place': 1},
'transaction.transaction': {'recipient': 1, 'giver': 1, 'beneficiary': 1, 'time': 1, 'place': 1},
'transaction.transfermoney': {'recipient': 1, 'giver': 1, 'money': 1, 'time': 1, 'beneficiary': 1, 'place': 1},
'transaction.transferownership': {'recipient': 1, 'thing': 1, 'giver': 1, 'place': 1, 'time': 1, 'beneficiary': 1},
}

# only 17 for ere17
KBP17_TYPES = {z.lower() for z in ['Contact.Broadcast', 'Transaction.TransferMoney', 'Movement.TransportPerson', 'Conflict.Attack', 'Transaction.TransferOwnership', 'Movement.TransportArtifact', 'Contact.Contact', 'Life.Die', 'Conflict.Demonstrate', 'Contact.Meet', 'Contact.Correspondence', 'Personnel.EndPosition', 'Personnel.Elect', 'Manufacture.Artifact', 'Personnel.StartPosition', 'Justice.ArrestJail', 'Life.Injure', 'Transaction.Transaction']}

# shortcut to obtain a copy!
def get_frames_label_budgets(key: str):
    ret = {"ace": ACE_ARG_BUDGETS, "ere": ERE_ARG_BUDGETS}[key]
    return ret.copy()
