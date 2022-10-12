#

# onto resources (pre-defined ones)

# --
# ace
# (using python3 -m msp2.tasks.zmtl3.scripts.misc.stat_and_modify input:../../events/data/data21f/en.ace2.train.json print_onto:1)
ONTO_ACE = {
'frames': [
{'name': 'Business:Declare-Bankruptcy', 'vp': 'declare bankruptcy', 'core_roles': ['Org', 'Place'], 'template': '<Org> declare bankruptcy in <Place>'},
{'name': 'Business:End-Org', 'vp': 'shut down', 'core_roles': ['Org', 'Place'], 'template': '<Org> shut down in <Place>'},
{'name': 'Business:Merge-Org', 'vp': 'merge', 'core_roles': ['Org', 'Place'], 'template': '<Org> merge in <Place>'},
{'name': 'Business:Start-Org', 'vp': 'start', 'core_roles': ['Agent', 'Org', 'Place'], 'template': '<Agent> start <Org> in <Place>'},
{'name': 'Conflict:Attack', 'vp': 'attack', 'core_roles': ['Attacker', 'Instrument', 'Place', 'Target'], 'template': '<Attacker> attack <Target> with <Instrument> in <Place>'},
{'name': 'Conflict:Demonstrate', 'vp': 'demonstrate', 'core_roles': ['Entity', 'Place'], 'template': '<Entity> demonstrate in <Place>'},
{'name': 'Contact:Meet', 'vp': 'meet', 'core_roles': ['Entity', 'Place'], 'template': '<Entity> meet in <Place>'},
{'name': 'Contact:Phone-Write', 'vp': 'communicate remotely', 'core_roles': ['Entity', 'Place'], 'template': '<Entity> communicate remotely in <Place>'},
{'name': 'Justice:Acquit', 'vp': 'acquit', 'core_roles': ['Adjudicator', 'Defendant', 'Place'], 'template': '<Adjudicator> acquit <Defendant> in <Place>'},
{'name': 'Justice:Appeal', 'vp': 'appeal', 'core_roles': ['Adjudicator', 'Place', 'Plaintiff'], 'template': '<Plaintiff> appeal to <Adjudicator> in <Place>'},
{'name': 'Justice:Arrest-Jail', 'vp': 'arrest', 'core_roles': ['Agent', 'Person', 'Place'], 'template': '<Agent> arrest <Person> in <Place>'},
{'name': 'Justice:Charge-Indict', 'vp': 'charge or indict', 'core_roles': ['Adjudicator', 'Defendant', 'Place', 'Prosecutor'], 'template': '<Prosecutor> charge or indict <Defendant> before <Adjudicator> in <Place>'},
{'name': 'Justice:Convict', 'vp': 'convict', 'core_roles': ['Adjudicator', 'Defendant', 'Place'], 'template': '<Adjudicator> convict <Defendant> in <Place>'},
{'name': 'Justice:Execute', 'vp': 'execute', 'core_roles': ['Agent', 'Person', 'Place'], 'template': '<Agent> execute <Person> in <Place>'},
{'name': 'Justice:Extradite', 'vp': 'extradite', 'core_roles': ['Agent', 'Destination', 'Origin', 'Person'], 'template': '<Agent> extradite <Person> from <Origin> to <Destination>'},
{'name': 'Justice:Fine', 'vp': 'fine', 'core_roles': ['Adjudicator', 'Entity', 'Place'], 'template': '<Adjudicator> fine <Entity> in <Place>'},
{'name': 'Justice:Pardon', 'vp': 'pardon', 'core_roles': ['Adjudicator', 'Defendant', 'Place'], 'template': '<Adjudicator> pardon <Defendant> in <Place>'},
{'name': 'Justice:Release-Parole', 'vp': 'release or parole', 'core_roles': ['Entity', 'Person', 'Place'], 'template': '<Entity> release or parole <Person> in <Place>'},
{'name': 'Justice:Sentence', 'vp': 'sentence', 'core_roles': ['Adjudicator', 'Defendant', 'Place'], 'template': '<Adjudicator> sentence <Defendant> in <Place>'},
{'name': 'Justice:Sue', 'vp': 'sue', 'core_roles': ['Adjudicator', 'Defendant', 'Place', 'Plaintiff'], 'template': '<Plaintiff> sue <Defendant> before <Adjudicator> in <Place>'},
{'name': 'Justice:Trial-Hearing', 'vp': 'try', 'core_roles': ['Adjudicator', 'Defendant', 'Place', 'Prosecutor'], 'template': '<Prosecutor> try <Defendant> before <Adjudicator> in <Place>'},
{'name': 'Life:Be-Born', 'vp': 'born', 'core_roles': ['Person', 'Place'], 'template': '<Person> born in <Place>'},
{'name': 'Life:Die', 'vp': 'kill', 'core_roles': ['Agent', 'Instrument', 'Place', 'Victim'], 'template': '<Agent> kill <Victim> with <Instrument> in <Place>'},
{'name': 'Life:Divorce', 'vp': 'divorce', 'core_roles': ['Person', 'Place'], 'template': '<Person> divorce in <Place>'},
{'name': 'Life:Injure', 'vp': 'injure', 'core_roles': ['Agent', 'Instrument', 'Place', 'Victim'], 'template': '<Agent> injure <Victim> with <Instrument> in <Place>'},
{'name': 'Life:Marry', 'vp': 'marry', 'core_roles': ['Person', 'Place'], 'template': '<Person> marry in <Place>'},
{'name': 'Movement:Transport', 'vp': 'transport', 'core_roles': ['Agent', 'Artifact', 'Destination', 'Origin', 'Vehicle'], 'template': '<Agent> transport <Artifact> with <Vehicle> from <Origin> to <Destination>'},
{'name': 'Personnel:Elect', 'vp': 'elect', 'core_roles': ['Entity', 'Person', 'Place'], 'template': '<Entity> elect <Person> in <Place>'},
{'name': 'Personnel:End-Position', 'vp': 'stop working', 'core_roles': ['Entity', 'Person', 'Place'], 'template': '<Person> stop working for <Entity> in <Place>'},
{'name': 'Personnel:Nominate', 'vp': 'nominate', 'core_roles': ['Agent', 'Person', 'Place'], 'template': '<Agent> nominate <Person> in <Place>'},
{'name': 'Personnel:Start-Position', 'vp': 'start working', 'core_roles': ['Entity', 'Person', 'Place'], 'template': '<Person> start working for <Entity> in <Place>'},
{'name': 'Transaction:Transfer-Money', 'vp': 'give money', 'core_roles': ['Beneficiary', 'Giver', 'Place', 'Recipient'], 'template': '<Giver> give money to <Recipient> for <Beneficiary> in <Place>'},
{'name': 'Transaction:Transfer-Ownership', 'vp': 'give', 'core_roles': ['Artifact', 'Beneficiary', 'Buyer', 'Place', 'Seller'], 'template': '<Seller> give <Artifact> to <Buyer> for <Beneficiary> in <Place>'},
],
'roles': [
{'name': 'Adjudicator', 'np': 'adjudicator', 'qwords': ['who']},
{'name': 'Agent', 'np': 'agent', 'qwords': ['who']},
{'name': 'Artifact', 'np': 'artifact'},
{'name': 'Attacker', 'np': 'attacker', 'qwords': ['who']},
{'name': 'Beneficiary', 'np': 'beneficiary', 'qwords': ['who']},
{'name': 'Buyer', 'np': 'buyer', 'qwords': ['who']},
{'name': 'Defendant', 'np': 'defendant', 'qwords': ['who']},
{'name': 'Destination', 'np': 'destination', 'qwords': ['where2']},
{'name': 'Entity', 'np': 'entity', 'qwords': ['who']},
{'name': 'Giver', 'np': 'giver', 'qwords': ['who']},
{'name': 'Instrument', 'np': 'instrument'},
{'name': 'Org', 'np': 'organization'},
{'name': 'Origin', 'np': 'origin', 'qwords': ['where2']},
{'name': 'Person', 'np': 'person', 'qwords': ['who']},
{'name': 'Place', 'np': 'place', 'qwords': ['where', 'where2']},
{'name': 'Plaintiff', 'np': 'plaintiff', 'qwords': ['who']},
{'name': 'Prosecutor', 'np': 'prosecutor', 'qwords': ['who']},
{'name': 'Recipient', 'np': 'recipient', 'qwords': ['who']},
{'name': 'Seller', 'np': 'seller', 'qwords': ['who']},
{'name': 'Target', 'np': 'target'},
{'name': 'Vehicle', 'np': 'vehicle'},
{'name': 'Victim', 'np': 'victim', 'qwords': ['who']},
],
}

# --
# ere
_DIFF_ERE = {
"del_frames": {'Contact:Phone-Write', 'Justice:Appeal', 'Justice:Release-Parole', 'Movement:Transport', 'Personnel:Elect', 'Transaction:Transfer-Ownership'},
"frames": [
{'name': 'Contact:Broadcast', 'vp': 'broadcast', 'core_roles': ['Audience', 'Entity', 'Place'], 'template': '<Entity> broadcast before <Audience> in <Place>'},
{'name': 'Contact:Contact', 'vp': 'communicate', 'core_roles': ['Entity', 'Place'], 'template': '<Entity> communicate in <Place>'},
{'name': 'Contact:Correspondence', 'vp': 'communicate remotely', 'core_roles': ['Entity', 'Place'], 'template': '<Entity> communicate remotely in <Place>'},
{'name': 'Justice:Appeal', 'vp': 'appeal', 'core_roles': ['Adjudicator', 'Defendant', 'Place'], 'template': '<Defendant> appeal to <Adjudicator> in <Place>'},
{'name': 'Justice:Release-Parole', 'vp': 'release or parole', 'core_roles': ['Agent', 'Person', 'Place'], 'template': '<Agent> release or parole <Person> in <Place>'},
{'name': 'Manufacture:Artifact', 'vp': 'manufacture', 'core_roles': ['Agent', 'Artifact', 'Place'], 'template': '<Agent> manufacture <Artifact> in <Place>'},
{'name': 'Movement:Transport-Artifact', 'vp': 'transport', 'core_roles': ['Agent', 'Artifact', 'Destination', 'Instrument', 'Origin'], 'template': '<Agent> transport <Artifact> with <Instrument> from <Origin> to <Destination>'},
{'name': 'Movement:Transport-Person', 'vp': 'transport', 'core_roles': ['Agent', 'Destination', 'Instrument', 'Origin', 'Person'], 'template': '<Agent> transport <Person> with <Instrument> from <Origin> to <Destination>'},
{'name': 'Personnel:Elect', 'vp': 'elect', 'core_roles': ['Agent', 'Person', 'Place'], 'template': '<Agent> elect <Person> in <Place>'},
{'name': 'Transaction:Transaction', 'vp': 'give something', 'core_roles': ['Beneficiary', 'Giver', 'Place', 'Recipient'], 'template': '<Giver> give something to <Recipient> for <Beneficiary> in <Place>'},
{'name': 'Transaction:Transfer-Ownership', 'vp': 'give', 'core_roles': ['Beneficiary', 'Giver', 'Place', 'Recipient', 'Thing'], 'template': '<Giver> give <Thing> to <Recipient> for <Beneficiary> in <Place>'},
],
"del_roles": {'Buyer', 'Seller', 'Vehicle'},
"roles": [
{'name': 'Audience', 'np': 'audience', 'qwords': ['who']},
{'name': 'Thing', 'np': 'thing'},
],
}
ONTO_ERE = {
'frames': sorted([z for z in ONTO_ACE['frames'] if z['name'] not in _DIFF_ERE['del_frames']]+_DIFF_ERE['frames'], key=lambda x: x['name']),
'roles': sorted([z for z in ONTO_ACE['roles'] if z['name'] not in _DIFF_ERE['del_roles']]+_DIFF_ERE['roles'], key=lambda x: x['name']),
}
ONTO_QA = {  # note: simply a dummy one!
'frames': [{'name': 'Q', 'vp': 'question', 'core_roles': ['A']}],
'roles': [{'name': 'A', 'np': 'answer'}]
}
# --
# specific questions
_QUES={'ace': {'be-born': {'Person': 'Who is born?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'marry': {'Person': 'Who is married?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'divorce': {'Person': 'Who is divorced?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'injure': {'Agent': 'Who injures someone?', 'Victim': 'Who is injured?', 'Instrument': 'What instrument is someone injured with?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'die': {'Agent': 'Who kills someone?', 'Victim': 'Who is killed?', 'Instrument': 'What instrument is someone killed with?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'transport': {'Agent': 'Who is responsible for the <T>?', 'Artifact': 'What is transported?', 'Vehicle': 'What is the vehicle used in the <T>?', 'Price': 'What is the cost of the <T>?', 'Origin': 'Where is the origin of the <T>?', 'Destination': 'Where is the destination of the <T>?', 'Time': 'When is the <T>?'}, 'transfer-ownership': {'Buyer': 'Who buys something?', 'Seller': 'Who sells anything?', 'Artifact': 'What is bought?', 'Price': 'How much does something cost?', 'Beneficiary': 'Who is something bought for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'transfer-money': {'Giver': 'Who gives the money?', 'Recipient': 'Who receives the money?', 'Beneficiary': 'Who is the beneficiary of the <T>?', 'Money': 'How much is given?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'start-org': {'Agent': 'Who starts an organization?', 'Org': 'What organization is started?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'merge-org': {'Org': 'What organization is merged?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'declare-bankruptcy': {'Org': 'What organization is bankrupt?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'end-org': {'Org': 'What organization is ended?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'attack': {'Attacker': 'Who attacks someone?', 'Target': 'Who is attacked?', 'Instrument': 'What instrument is someone attacked with?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'demonstrate': {'Entity': 'Who goes on a demonstration?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'meet': {'Entity': 'Who meets with someone?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'phone-write': {'Entity': 'Who has a written or phone communication with someone?', 'Time': 'When is the <T>?'}, 'start-position': {'Person': 'Who starts a job?', 'Entity': 'Who hires someone?', 'Position': 'What is the new position?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'end-position': {'Person': 'Who ends a job?', 'Entity': 'What organization does someone leave from?', 'Position': 'What position does someone leave?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'nominate': {'Agent': 'Who nominates someone?', 'Person': 'Who is nominated?', 'Position': 'What is someone nominated as?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'elect': {'Entity': 'Who elects someone?', 'Person': 'Who is elected?', 'Position': 'What is someone elected as?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'arrest-jail': {'Agent': 'Who arrests someone?', 'Person': 'Who is arrested?', 'Crime': 'What crime is someone arrested for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'release-parole': {'Agent': 'Who releases someone?', 'Person': 'Who is released?', 'Crime': 'What crime is someone previously being held for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'trial-hearing': {'Prosecutor': 'Who is the prosecutor?', 'Defendant': 'Who is on trial?', 'Crime': 'What crime is someone being tried for?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'charge-indict': {'Prosecutor': 'Who is the prosecutor?', 'Defendant': 'Who is charged?', 'Crime': 'What crime is someone charged for?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'sue': {'Plaintiff': 'Who sues someone?', 'Defendant': 'Who is sued?', 'Crime': 'What crime is someone sued for?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'convict': {'Defendant': 'Who is convicted?', 'Crime': 'What crime is someone convicted for?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'sentence': {'Defendant': 'Who is sentenced?', 'Crime': 'What crime is someone convicted for?', 'Sentence': 'What is the sentence?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'fine': {'Adjudicator': 'Who fines someone?', 'Entity': 'Who is fined?', 'Crime': 'What crime is someone fined for?', 'Money': 'How much is the fine?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'execute': {'Agent': 'Who executes someone?', 'Person': 'Who is executed?', 'Crime': 'What crime is someone executed for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'extradite': {'Agent': 'Who extradites someone?', 'Person': 'Who is extradited?', 'Crime': 'What crime is someone extradited for?', 'Origin': 'Where is the origin of the extradition?', 'Destination': 'Where is the destination of the extradition?', 'Time': 'When is the <T>?'}, 'acquit': {'Adjudicator': 'Who acquits someone?', 'Defendant': 'Who is acquitted?', 'Crime': 'What crime is someone previously charged for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'pardon': {'Adjudicator': 'Who pardons someone?', 'Defendant': 'Who is pardoned?', 'Crime': 'What crime is someone pardoned for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'appeal': {'Defendant': 'Who makes an appeal?', 'Prosecutor': 'Who is the prosecutor?', 'Crime': 'What is the crime?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}}, 'ere': {'be-born': {'Person': 'Who is born?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'marry': {'Person': 'Who is married?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'divorce': {'Person': 'Who is divorced?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'injure': {'Agent': 'Who injures someone?', 'Victim': 'Who is injured?', 'Instrument': 'What instrument is someone injured with?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'die': {'Agent': 'Who kills someone?', 'Victim': 'Who is killed?', 'Instrument': 'What instrument is someone killed with?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'transport-person': {'Agent': 'Who is responsible for the <T>?', 'Person': 'What person is transported?', 'Instrument': 'What is the vehicle used in the <T>?', 'Origin': 'Where is the origin of the <T>?', 'Destination': 'Where is the destination of the <T>?', 'Time': 'When is the <T>?'}, 'transport-artifact': {'Agent': 'Who is responsible for the <T>?', 'Artifact': 'What thing is transported?', 'Instrument': 'What is the vehicle used in the <T>?', 'Origin': 'Where is the origin of the <T>?', 'Destination': 'Where is the destination of the <T>?', 'Time': 'When is the <T>?'}, 'transfer-ownership': {'Recipient': 'Who buys something?', 'Giver': 'Who sells anything?', 'Thing': 'What is bought?', 'Beneficiary': 'Who is something bought for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'transfer-money': {'Giver': 'Who gives the money?', 'Recipient': 'Who receives the money?', 'Beneficiary': 'Who is the beneficiary of the <T>?', 'Money': 'How much is given?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'transaction': {'Giver': 'Who gives something?', 'Recipient': 'Who receives the something?', 'Beneficiary': 'Who is the beneficiary of the <T>?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'start-org': {'Agent': 'Who starts an organization?', 'Org': 'What organization is started?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'merge-org': {'Agent': 'Who merges an organization?', 'Org': 'What organization is merged?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'declare-bankruptcy': {'Org': 'What organization is bankrupt?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'end-org': {'Org': 'What organization is ended?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'attack': {'Attacker': 'Who attacks someone?', 'Target': 'Who is attacked?', 'Instrument': 'What instrument is someone attacked with?', 'Place': 'Where is the <T>?', 'Time': 'When is the <T>?'}, 'demonstrate': {'Entity': 'Who goes on a demonstration?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'meet': {'Entity': 'Who meets with someone?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'correspondence': {'Entity': 'Who corresponds with someone?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'broadcast': {'Entity': 'Who broadcasts something?', 'Audience': 'Who is the audience of the broadcast?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'contact': {'Entity': "Who contacts with someone, but unclear if it's face-to-face?", 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'artifact': {'Agent': 'Who manufactures something?', 'Artifact': 'What is manufactured?', 'Instrument': 'What instrument is used in the <T>?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'start-position': {'Person': 'Who starts a job?', 'Entity': 'Who hires someone?', 'Position': 'What is the new position?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'end-position': {'Person': 'Who ends a job?', 'Entity': 'What organization does someone leave from?', 'Position': 'What position does someone leave?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'nominate': {'Agent': 'Who nominates someone?', 'Person': 'Who is nominated?', 'Position': 'What is someone nominated as?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'elect': {'Agent': 'Who elects someone?', 'Person': 'Who is elected?', 'Position': 'What is someone elected as?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'arrest-jail': {'Agent': 'Who arrests someone?', 'Person': 'Who is arrested?', 'Crime': 'What crime is someone arrested for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'release-parole': {'Agent': 'Who releases someone?', 'Person': 'Who is released?', 'Crime': 'What crime is someone previously being held for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'trial-hearing': {'Prosecutor': 'Who is the prosecutor?', 'Defendant': 'Who is on trial?', 'Crime': 'What crime is someone being tried for?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'charge-indict': {'Prosecutor': 'Who is the prosecutor?', 'Defendant': 'Who is charged?', 'Crime': 'What crime is someone charged for?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'sue': {'Plaintiff': 'Who sues someone?', 'Defendant': 'Who is sued?', 'Crime': 'What crime is someone sued for?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'convict': {'Defendant': 'Who is convicted?', 'Crime': 'What crime is someone convicted for?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'sentence': {'Defendant': 'Who is sentenced?', 'Crime': 'What crime is someone convicted for?', 'Sentence': 'What is the sentence?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'fine': {'Adjudicator': 'Who fines someone?', 'Entity': 'Who is fined?', 'Crime': 'What crime is someone fined for?', 'Money': 'How much is the fine?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'execute': {'Agent': 'Who executes someone?', 'Person': 'Who is executed?', 'Crime': 'What crime is someone executed for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'extradite': {'Agent': 'Who extradites someone?', 'Person': 'Who is extradited?', 'Crime': 'What crime is someone extradited for?', 'Origin': 'Where is the origin of the extradition?', 'Destination': 'Where is the destination of the extradition?', 'Time': 'When is the <T>?'}, 'acquit': {'Adjudicator': 'Who acquits someone?', 'Defendant': 'Who is acquitted?', 'Crime': 'What crime is someone previously charged for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'pardon': {'Adjudicator': 'Who pardons someone?', 'Defendant': 'Who is pardoned?', 'Crime': 'What crime is someone pardoned for?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}, 'appeal': {'Defendant': 'Who makes an appeal?', 'Prosecutor': 'Who is the prosecutor?', 'Crime': 'What is the crime?', 'Adjudicator': 'Who is the adjudicator?', 'Time': 'When is the <T>?', 'Place': 'Where is the <T>?'}}}
# --

# --
def get_predefined_onto(name: str):
    # ace: Onto:frames=33,roles=22,core=102
    # ere: Onto:frames=38,roles=21,core=119
    # rams: Onto:frames=139,roles=65,core=506
    # --
    from .onto_rams import ONTO_RAMS, ONTO_BIO11
    if name == 'pbfn':
        import os
        from msp2.utils import default_json_serializer
        # read from local
        ret = default_json_serializer.from_file(os.path.join(os.path.dirname(__file__), 'onto_pbfn.json.gz'))
    else:
        d = {'ace': ONTO_ACE, 'ere': ONTO_ERE, 'qa': ONTO_QA, 'rams': ONTO_RAMS, 'bio11': ONTO_BIO11}
        ret = d.get(name)
        if ret is None and name[-1:]=='T' and name[:-1] in d:
            ret = d[name[:-1]]
            # --
            # parse the templates
            for vv in ret['frames']:
                _tpl0 = vv['template']
                assert vv['vp'] in _tpl0
                _fs = _tpl0.replace(vv['vp'], "<>").split()
                _tpl1 = []
                _tmp_s = ""
                for _ff in _fs:
                    if _ff == "<>":
                        _tpl1.append([None, []])
                        _tmp_s = ""
                    elif _ff.startswith("<") and _ff.endswith(">"):
                        _tpl1.append([_ff[1:-1], ([] if _tmp_s=="" else [_tmp_s])])
                        _tmp_s = ""
                    else:
                        _tmp_s = _ff if (_tmp_s=="") else f"{_tmp_s} {_ff}"
                vv['template'] = _tpl1
            # breakpoint()
            # --
    if ret is not None and name in _QUES:
        ques = _QUES[name]
        for vv in ret['frames']:
            nn = vv['name'].split(":")[-1].lower()
            qq = ques[nn]
            if 'Place' not in qq:
                qq['Place'] = "Where is the <T>?"
            # mismatch between data & onto
            if nn=='appeal' and 'Plantiff' not in qq:
                qq['Plaintiff'] = 'Who makes an appeal?'
            if nn=='release-parole' and 'Entity' not in qq:
                qq['Entity'] = 'Who releases someone?'
            # --
            role_questions = {k: qq[k] for k in vv['core_roles']}
            vv['role_questions'] = role_questions
    # --
    return ret
# --
