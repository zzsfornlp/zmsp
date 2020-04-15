#

# =====
# normalization for the labels

from typing import List, Dict
from msp.zext.ie import rearrange_camel

# =====
# normalize labels for different dataset

def norm_label(s: str, m: Dict = None):
    if m is None:
        m = {}
    ss = [m.get(x, x) for x in s.split(".")]
    return ".".join([rearrange_camel(x) for x in ss])

def get_label_normer(dataset_id: str):
    dataset_id = str.lower(dataset_id)
    if dataset_id.startswith("ace"):
        return ACENormer()
    elif dataset_id.startswith("ldc") and int(dataset_id[3:7])<2018:
        return ERENormer()
    else:
        return OthersNormer()
        # raise NotImplementedError(dataset_id)

# =====

# does not do anything
class OthersNormer:
    def norm_evt_label(self, s: str):
        return s

    def norm_role_label(self, s: str):
        return s

    def norm_ef_label(self, s: str):
        return s

class ACENormer:
    EVT_MAP = {"Phone-Write": "Correspondence", "Be-Born": "Born"}
    EF_MAP = {"E-Mail": "Email", "County-or-District": "CountyDistrict", "GPE-Cluster": "Cluster", "State-or-Province": "StateProvince", "Non-Governmental": "Nongovernmental"}

    def norm_evt_label(self, s: str):
        return norm_label(s, ACENormer.EVT_MAP)

    def norm_role_label(self, s: str):
        if s.startswith("Time"):
            s = ".".join(s.split("-", 1))  # make it secondary type
        return norm_label(s)

    def norm_ef_label(self, s: str):
        return norm_label(s, ACENormer.EF_MAP)

class ERENormer:
    EVT_MAP = {"declarebankruptcy": "DeclareBankruptcy", "endorg": "EndOrg", "mergeorg": "MergeOrg", "startorg": "StartOrg", "arrestjail": "ArrestJail", "chargeindict": "ChargeIndict", "releaseparole": "ReleaseParole", "trialhearing": "TrialHearing", "beborn": "Born", "transportartifact": "TransportArtifact", "transportperson": "TransportPerson", "endposition": "EndPosition", "startposition": "StartPosition", "transfermoney": "TransferMoney", "transferownership": "TransferOwnership"}
    EF_MAP = {"vehicle": "VEH", "weapon": "WEA"}

    def norm_evt_label(self, s: str):
        return norm_label(s, ERENormer.EVT_MAP)

    def norm_role_label(self, s: str):
        return norm_label(s)

    def norm_ef_label(self, s: str):
        return norm_label(s, ERENormer.EF_MAP)

#
KBP17_TYPES = {'Contact.Broadcast', 'Transaction.TransferMoney', 'Movement.TransportPerson', 'Conflict.Attack', 'Transaction.TransferOwnership', 'Movement.TransportArtifact', 'Contact.Contact', 'Life.Die', 'Conflict.Demonstrate', 'Contact.Meet', 'Contact.Correspondence', 'Personnel.EndPosition', 'Personnel.Elect', 'Manufacture.Artifact', 'Personnel.StartPosition', 'Justice.ArrestJail', 'Life.Injure', 'Transaction.Transaction'}

# =====
# argument constraints

# -----
# for ere: these are read from mannual
# basic budgets (time and place should be added to all)
ERE_ARG_BUDGETS = {
    "Business.DeclareBankruptcy": {"Org": 1},
    "Business.EndOrg": {"Org": 1},
    "Business.MergeOrg": {"Agent": 1, "Org": 1},
    "Business.StartOrg": {"Agent": 1, "Org": 1},
    "Conflict.Attack": {"Attacker": 1, "Target": 1, "Instrument": 1},
    "Conflict.Demonstrate": {"Entity": 1},
    "Contact.Broadcast": {"Entity": 1, "Audience": 1},
    "Contact.Contact": {"Entity": 2},
    "Contact.Correspondence": {"Entity": 2},
    "Contact.Meet": {"Entity": 2},
    "Justice.Acquit": {"Adjudicator": 1, "Defendant": 1, "Crime": 1},
    "Justice.Appeal": {"Prosecutor": 1, "Adjudicator": 1, "Defendant": 1, "Crime": 1},
    "Justice.ArrestJail": {"Agent": 1, "Person": 1, "Crime": 1},
    "Justice.ChargeIndict": {"Prosecutor": 1, "Adjudicator": 1, "Defendant": 1, "Crime": 1},
    "Justice.Convict": {"Adjudicator": 1, "Defendant": 1, "Crime": 1},
    "Justice.Execute": {"Agent": 1, "Person": 1, "Crime": 1},
    "Justice.Extradite": {"Agent": 1, "Person": 1, "Crime": 1, "Origin": 1, "Destination": 1},
    "Justice.Fine": {"Adjudicator": 1, "Entity": 1, "Crime": 1, "Money": 1},
    "Justice.Pardon": {"Adjudicator": 1, "Defendant": 1, "Crime": 1},
    "Justice.ReleaseParole": {"Agent": 1, "Person": 1, "Crime": 1},
    "Justice.Sentence": {"Adjudicator": 1, "Defendant": 1, "Crime": 1, "Sentence": 1},
    "Justice.Sue": {"Plaintiff": 1, "Adjudicator": 1, "Defendant": 1, "Crime": 1},
    "Justice.TrialHearing": {"Prosecutor": 1, "Adjudicator": 1, "Defendant": 1, "Crime": 1},
    "Life.Born": {"Person": 1},
    "Life.Die": {"Agent": 1, "Victim": 1, "Instrument": 1},
    "Life.Divorce": {"Person": 2},
    "Life.Injure": {"Agent": 1, "Victim": 1, "Instrument": 1},
    "Life.Marry": {"Person": 2},
    "Manufacture.Artifact": {"Agent": 1, "Artifact": 1, "Instrument": 1},
    "Movement.TransportArtifact": {"Agent": 1, "Artifact": 1, "Instrument": 1, "Origin": 1, "Destination": 1},
    "Movement.TransportPerson": {"Agent": 1, "Person": 1, "Instrument": 1, "Origin": 1, "Destination": 1},
    "Personnel.Elect": {"Agent": 1, "Person": 1, "Position": 1},
    "Personnel.EndPosition": {"Entity": 1, "Person": 1, "Position": 1},
    "Personnel.Nominate": {"Agent": 1, "Person": 1, "Position": 1},
    "Personnel.StartPosition": {"Entity": 1, "Person": 1, "Position": 1},
    "Transaction.Transaction": {"Giver": 1, "Recipient": 1, "Beneficiary": 1},
    "Transaction.TransferMoney": {"Giver": 1, "Recipient": 1, "Beneficiary": 1, "Money": 1},
    "Transaction.TransferOwnership": {"Giver": 1, "Recipient": 1, "Beneficiary": 1, "Thing": 1},
}
for k, v in ERE_ARG_BUDGETS.items():
    if "Origin" in v or "Destination" in v:
        assert "Origin" in v and "Destination" in v  # no Place for moving ones
        v.update({"Time": 1})
    else:
        v.update({"Time": 1, "Place": 1})

# -----
# for ace: these are read from data
ACE_ARG_BUDGETS = {
"Business.DeclareBankruptcy": {'Org': 1, 'Time.After': 1, 'Place': 1, 'Time.Within': 1, 'Time.AtBeginning': 1},
"Business.EndOrg": {'Org': 1, 'Time.Holds': 1, 'Place': 1, 'Time.Within': 1, 'Time.After': 1},
"Business.MergeOrg": {'Org': 1, 'Time.Ending': 1},
"Business.StartOrg": {'Org': 1, 'Agent': 1, 'Place': 1, 'Time.Within': 1, 'Time.Starting': 1, 'Time.Before': 1},
"Conflict.Attack": {'Attacker': 1, 'Place': 1, 'Target': 1, 'Time.Within': 1, 'Time.Ending': 1, 'Instrument': 1, 'Time.Holds': 1, 'Time.AtBeginning': 1, 'Victim': 1, 'Time.After': 1, 'Time.Starting': 1, 'Time.Before': 1, 'Time.AtEnd': 1},
"Conflict.Demonstrate": {'Place': 1, 'Entity': 1, 'Time.Within': 1, 'Time.AtEnd': 1, 'Time.Starting': 1},
"Contact.Correspondence": {'Entity': 2, 'Time.Within': 1, 'Time.Before': 1, 'Place': 1, 'Time.Holds': 1, 'Time.Starting': 1},
"Contact.Meet": {'Place': 1, 'Entity': 2, 'Time.Within': 1, 'Time.Holds': 1, 'Time.AtBeginning': 1, 'Time.Starting': 1, 'Time.Ending': 1, 'Time.After': 1},
"Justice.Acquit": {'Defendant': 1, 'Adjudicator': 1, 'Crime': 1, 'Time.Within': 1},
"Justice.Appeal": {'Adjudicator': 1, 'Plaintiff': 1, 'Place': 1, 'Crime': 1, 'Time.Within': 1},
"Justice.ArrestJail": {'Person': 1, 'Agent': 1, 'Time.Within': 1, 'Place': 1, 'Crime': 1, 'Time.Holds': 1, 'Time.Ending': 1, 'Time.AtBeginning': 1, 'Time.Starting': 1, 'Time.Before': 1},
"Justice.ChargeIndict": {'Adjudicator': 1, 'Defendant': 1, 'Crime': 1, 'Prosecutor': 1, 'Time.Within': 1, 'Place': 1, 'Time.Before': 1},
"Justice.Convict": {'Defendant': 1, 'Crime': 1, 'Adjudicator': 1, 'Time.Within': 1, 'Place': 1, 'Time.AtBeginning': 1},
"Justice.Execute": {'Agent': 1, 'Place': 1, 'Person': 1, 'Time.Within': 1, 'Time.After': 1, 'Crime': 1, 'Time.AtBeginning': 1},
"Justice.Extradite": {'Destination': 1, 'Origin': 1, 'Person': 1, 'Agent': 1},
"Justice.Fine": {'Entity': 1, 'Money': 1, 'Place': 1, 'Time.Within': 1, 'Crime': 1, 'Adjudicator': 1},
"Justice.Pardon": {'Defendant': 1, 'Adjudicator': 1, 'Place': 1, 'Time.AtEnd': 1},
"Justice.ReleaseParole": {'Person': 1, 'Entity': 1, 'Time.Within': 1, 'Place': 1, 'Crime': 1, 'Time.After': 1},
"Justice.Sentence": {'Adjudicator': 1, 'Defendant': 1, 'Sentence': 1, 'Crime': 1, 'Place': 1, 'Time.Within': 1, 'Time.Starting': 1},
"Justice.Sue": {'Plaintiff': 1, 'Defendant': 1, 'Place': 1, 'Crime': 1, 'Adjudicator': 1, 'Time.Within': 1, 'Time.Holds': 1},
"Justice.TrialHearing": {'Prosecutor': 1, 'Place': 1, 'Time.Within': 1, 'Defendant': 1, 'Crime': 1, 'Adjudicator': 1, 'Time.Starting': 1, 'Time.AtEnd': 1, 'Time.Holds': 1},
"Life.Born": {'Place': 1, 'Person': 1, 'Time.Within': 1, 'Time.Holds': 1},
"Life.Die": {'Victim': 1, 'Time.Within': 1, 'Agent': 1, 'Place': 1, 'Instrument': 1, 'Time.Before': 1, 'Time.After': 1, 'Person': 1, 'Time.Starting': 1, 'Time.Holds': 1, 'Time.Ending': 1, 'Time.AtBeginning': 1},
"Life.Divorce": {'Person': 2, 'Time.Within': 1},
"Life.Injure": {'Victim': 1, 'Agent': 1, 'Place': 1, 'Time.Within': 1, 'Instrument': 1},
"Life.Marry": {'Person': 2, 'Time.Within': 1, 'Place': 1, 'Time.Holds': 1, 'Time.Before': 1},
"Movement.Transport": {'Vehicle': 1, 'Artifact': 1, 'Destination': 1, 'Agent': 1, 'Time.AtBeginning': 1, 'Time.Within': 1, 'Origin': 1, 'Time.AtEnd': 1, 'Time.Starting': 1, 'Time.Ending': 1, 'Time.After': 1, 'Time.Holds': 1, 'Victim': 1, 'Place': 1, 'Time.Before': 1},
"Personnel.Elect": {'Person': 1, 'Position': 1, 'Entity': 1, 'Place': 1, 'Time.Within': 1, 'Time.Starting': 1, 'Time.Holds': 1, 'Time.Before': 1, 'Time.AtBeginning': 1},
"Personnel.EndPosition": {'Person': 1, 'Entity': 1, 'Position': 1, 'Time.Within': 1, 'Place': 1, 'Time.Ending': 1, 'Time.Holds': 1, 'Time.Before': 1, 'Time.Starting': 1, 'Time.After': 1},
"Personnel.Nominate": {'Person': 1, 'Agent': 1, 'Time.Within': 1, 'Position': 1},
"Personnel.StartPosition": {'Person': 1, 'Entity': 1, 'Position': 1, 'Place': 1, 'Time.Within': 1, 'Time.AtBeginning': 1, 'Time.After': 1, 'Time.Starting': 1, 'Time.Holds': 1},
"Transaction.TransferMoney": {'Giver': 1, 'Recipient': 1, 'Money': 1, 'Beneficiary': 1, 'Time.Holds': 1, 'Time.Within': 1, 'Time.Starting': 1, 'Place': 1, 'Time.Before': 1},
"Transaction.TransferOwnership": {'Artifact': 1, 'Buyer': 1, 'Seller': 1, 'Time.Within': 1, 'Place': 1, 'Time.Before': 1, 'Beneficiary': 1, 'Price': 1, 'Time.AtBeginning': 1},
}

# =====
# aida-v1 from RAMS

# =====
# reader
def read_budget():
    for line in open("scorer/event_role_multiplicities.txt"):
        fields = line.split()
        budgets = {fields[i]: int(fields[i+1]) for i in range(1, len(fields), 2)}
        print(f"'{fields[0]}': {budgets},")
# =====

RAMS_ARG_BUDGETS = {
'artifactexistence.damagedestroy.n/a': {'damagerdestroyer': 1, 'artifact': 1, 'instrument': 1, 'place': 1},
'artifactexistence.damagedestroy.damage': {'damager': 1, 'artifact': 1, 'instrument': 1, 'place': 1},
'artifactexistence.damagedestroy.destroy': {'destroyer': 1, 'artifact': 1, 'instrument': 1, 'place': 1},
'conflict.attack.n/a': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.airstrikemissilestrike': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.biologicalchemicalpoisonattack': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.bombing': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.firearmattack': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.hanging': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.invade': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.selfdirectedbattle': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.setfire': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.stabbing': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.attack.stealrobhijack': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1, 'artifact': 1},
'conflict.attack.strangling': {'attacker': 1, 'target': 1, 'instrument': 1, 'place': 1},
'conflict.demonstrate.n/a': {'demonstrator': 1, 'place': 1},
'conflict.demonstrate.marchprotestpoliticalgathering': {'demonstrator': 1, 'place': 1},
'conflict.yield.n/a': {'yielder': 1, 'recipient': 1, 'place': 1},
'conflict.yield.retreat': {'retreater': 1, 'origin': 1, 'destination': 1},
'conflict.yield.surrender': {'surrenderer': 1, 'recipient': 1, 'place': 1},
'contact.collaborate.n/a': {'participant': 2, 'place': 1},
'contact.collaborate.correspondence': {'participant': 2, 'place': 1},
'contact.collaborate.meet': {'participant': 2, 'place': 1},
'contact.commandorder.n/a': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.commandorder.broadcast': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.commandorder.correspondence': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.commandorder.meet': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.commitmentpromiseexpressintent.n/a': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.commitmentpromiseexpressintent.broadcast': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.commitmentpromiseexpressintent.correspondence': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.commitmentpromiseexpressintent.meet': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.discussion.n/a': {'participant': 2, 'place': 1},
'contact.discussion.correspondence': {'participant': 2, 'place': 1},
'contact.discussion.meet': {'participant': 2, 'place': 1},
'contact.funeralvigil.n/a': {'participant': 2, 'deceased': 1, 'place': 1},
'contact.funeralvigil.meet': {'participant': 2, 'deceased': 1, 'place': 1},
'contact.mediastatement.n/a': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.mediastatement.broadcast': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.negotiate.n/a': {'participant': 2, 'place': 1},
'contact.negotiate.correspondence': {'participant': 2, 'place': 1},
'contact.negotiate.meet': {'participant': 2, 'place': 1},
'contact.prevarication.n/a': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.prevarication.broadcast': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.prevarication.correspondence': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.prevarication.meet': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.publicstatementinperson.n/a': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.publicstatementinperson.broadcast': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.requestadvise.n/a': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.requestadvise.broadcast': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.requestadvise.correspondence': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.requestadvise.meet': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.threatencoerce.n/a': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.threatencoerce.broadcast': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.threatencoerce.correspondence': {'communicator': 1, 'recipient': 1, 'place': 1},
'contact.threatencoerce.meet': {'communicator': 1, 'recipient': 1, 'place': 1},
'disaster.accidentcrash.accidentcrash': {'driverpassenger': 1, 'vehicle': 1, 'crashobject': 1, 'place': 1},
'disaster.fireexplosion.fireexplosion': {'fireexplosionobject': 1, 'instrument': 1, 'place': 1},
'government.agreements.n/a': {'participant': 2, 'place': 1},
'government.agreements.acceptagreementcontractceasefire': {'participant': 2, 'place': 1},
'government.agreements.rejectnullifyagreementcontractceasefire': {'rejecternullifier': 1, 'otherparticipant': 1, 'place': 1},
'government.agreements.violateagreement': {'violator': 1, 'otherparticipant': 1, 'place': 1},
'government.formation.n/a': {'gpe': 1, 'founder': 1, 'place': 1},
'government.formation.mergegpe': {'participant': 2, 'place': 1},
'government.formation.startgpe': {'gpe': 1, 'founder': 1, 'place': 1},
'government.legislate.legislate': {'governmentbody': 1, 'law': 1, 'place': 1},
'government.spy.spy': {'spy': 1, 'observedentity': 1, 'beneficiary': 1, 'place': 1},
'government.vote.n/a': {'voter': 1, 'candidate': 1, 'ballot': 1, 'result': 1, 'place': 1},
'government.vote.castvote': {'voter': 1, 'candidate': 1, 'ballot': 1, 'result': 1, 'place': 1},
'government.vote.violationspreventvote': {'preventer': 1, 'voter': 1, 'candidate': 1, 'ballot': 1, 'place': 1},
'inspection.sensoryobserve.n/a': {'observer': 1, 'observedentity': 1, 'place': 1},
'inspection.sensoryobserve.inspectpeopleorganization': {'inspector': 1, 'inspectedentity': 1, 'place': 1},
'inspection.sensoryobserve.monitorelection': {'monitor': 1, 'monitoredentity': 1, 'place': 1},
'inspection.sensoryobserve.physicalinvestigateinspect': {'inspector': 1, 'inspectedentity': 1, 'place': 1},
'justice.arrestjaildetain.arrestjaildetain': {'jailer': 1, 'detainee': 1, 'crime': 1, 'place': 1},
'justice.initiatejudicialprocess.n/a': {'prosecutor': 1, 'defendant': 1, 'judgecourt': 1, 'crime': 1, 'place': 1},
'justice.initiatejudicialprocess.chargeindict': {'prosecutor': 1, 'defendant': 1, 'judgecourt': 1, 'crime': 1, 'place': 1},
'justice.initiatejudicialprocess.trialhearing': {'prosecutor': 1, 'defendant': 1, 'judgecourt': 1, 'crime': 1, 'place': 1},
'justice.investigate.n/a': {'investigator': 1, 'defendant': 1, 'place': 1},
'justice.investigate.investigatecrime': {'investigator': 1, 'defendant': 1, 'crime': 1, 'place': 1},
'justice.judicialconsequences.n/a': {'judgecourt': 1, 'defendant': 1, 'crime': 1, 'place': 1},
'justice.judicialconsequences.convict': {'judgecourt': 1, 'defendant': 1, 'crime': 1, 'place': 1},
'justice.judicialconsequences.execute': {'executioner': 1, 'defendant': 1, 'crime': 1, 'place': 1},
'justice.judicialconsequences.extradite': {'extraditer': 1, 'defendant': 1, 'crime': 1, 'origin': 1, 'destination': 1},
'life.die.n/a': {'victim': 1, 'place': 1},
'life.die.deathcausedbyviolentevents': {'killer': 1, 'victim': 1, 'instrument': 1, 'place': 1},
'life.die.nonviolentdeath': {'victim': 1, 'place': 1},
'life.injure.n/a': {'victim': 1, 'injurer': 1, 'place': 1},
'life.injure.illnessdegradationhungerthirst': {'victim': 1, 'place': 1},
'life.injure.illnessdegradationphysical': {'victim': 1},
'life.injure.injurycausedbyviolentevents': {'injurer': 1, 'victim': 1, 'instrument': 1, 'place': 1},
'manufacture.artifact.n/a': {'manufacturer': 1, 'artifact': 1, 'instrument': 1, 'place': 1},
'manufacture.artifact.build': {'manufacturer': 1, 'artifact': 1, 'instrument': 1, 'place': 1},
'manufacture.artifact.createintellectualproperty': {'manufacturer': 1, 'artifact': 1, 'instrument': 1, 'place': 1},
'manufacture.artifact.createmanufacture': {'manufacturer': 1, 'artifact': 1, 'instrument': 1, 'place': 1},
'movement.transportartifact.n/a': {'transporter': 1, 'artifact': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.bringcarryunload': {'transporter': 1, 'artifact': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.disperseseparate': {'transporter': 1, 'artifact': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.fall': {'artifact': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.grantentry': {'transporter': 1, 'artifact': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.hide': {'transporter': 1, 'artifact': 1, 'hidingplace': 1, 'vehicle': 1, 'origin': 1},
'movement.transportartifact.nonviolentthrowlaunch': {'transporter': 1, 'artifact': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.prevententry': {'preventer': 1, 'transporter': 1, 'artifact': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.preventexit': {'preventer': 1, 'transporter': 1, 'artifact': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.receiveimport': {'transporter': 1, 'artifact': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.sendsupplyexport': {'transporter': 1, 'artifact': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportartifact.smuggleextract': {'transporter': 1, 'artifact': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.n/a': {'transporter': 1, 'passenger': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.bringcarryunload': {'transporter': 1, 'passenger': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.disperseseparate': {'transporter': 1, 'passenger': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.evacuationrescue': {'transporter': 1, 'passenger': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.fall': {'passenger': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.grantentryasylum': {'granter': 1, 'transporter': 1, 'passenger': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.hide': {'transporter': 1, 'passenger': 1, 'hidingplace': 1, 'vehicle': 1, 'origin': 1},
'movement.transportperson.prevententry': {'preventer': 1, 'transporter': 1, 'passenger': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.preventexit': {'preventer': 1, 'transporter': 1, 'passenger': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.selfmotion': {'transporter': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'movement.transportperson.smuggleextract': {'transporter': 1, 'passenger': 1, 'vehicle': 1, 'origin': 1, 'destination': 1},
'personnel.elect.n/a': {'voter': 1, 'candidate': 1, 'place': 1},
'personnel.elect.winelection': {'voter': 1, 'candidate': 1, 'place': 1},
'personnel.endposition.n/a': {'employee': 1, 'placeofemployment': 1, 'place': 1},
'personnel.endposition.firinglayoff': {'employee': 1, 'placeofemployment': 1, 'place': 1},
'personnel.endposition.quitretire': {'employee': 1, 'placeofemployment': 1, 'place': 1},
'personnel.startposition.n/a': {'employee': 1, 'placeofemployment': 1, 'place': 1},
'personnel.startposition.hiring': {'employee': 1, 'placeofemployment': 1, 'place': 1},
'transaction.transaction.n/a': {'participant': 2, 'beneficiary': 1, 'place': 1},
'transaction.transaction.embargosanction': {'preventer': 1, 'giver': 1, 'recipient': 1, 'artifactmoney': 1, 'place': 1},
'transaction.transaction.giftgrantprovideaid': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'place': 1},
'transaction.transfermoney.n/a': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'money': 1, 'place': 1},
'transaction.transfermoney.borrowlend': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'money': 1, 'place': 1},
'transaction.transfermoney.embargosanction': {'preventer': 1, 'giver': 1, 'recipient': 1, 'money': 1, 'place': 1},
'transaction.transfermoney.giftgrantprovideaid': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'money': 1, 'place': 1},
'transaction.transfermoney.payforservice': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'money': 1, 'place': 1},
'transaction.transfermoney.purchase': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'money': 1, 'place': 1},
'transaction.transferownership.n/a': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'artifact': 1, 'place': 1},
'transaction.transferownership.borrowlend': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'artifact': 1, 'place': 1},
'transaction.transferownership.embargosanction': {'preventer': 1, 'giver': 1, 'recipient': 1, 'artifact': 1, 'place': 1},
'transaction.transferownership.giftgrantprovideaid': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'artifact': 1, 'place': 1},
'transaction.transferownership.purchase': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'artifact': 1, 'place': 1},
'transaction.transaction.transfercontrol': {'giver': 1, 'recipient': 1, 'beneficiary': 1, 'territoryorfacility': 1, 'place': 1},
}
# special for L2 type with n/a, union for L1 type
_RAMS_ITEMS = list(RAMS_ARG_BUDGETS.items())
for k, v in _RAMS_ITEMS:
    for rr in [1,2]:
        low_type = ".".join(k.split(".")[:rr])
        if low_type not in RAMS_ARG_BUDGETS:
            RAMS_ARG_BUDGETS[low_type] = v.copy()
        else:
            for r, c in v.items():
                RAMS_ARG_BUDGETS[low_type][r] = max(c, RAMS_ARG_BUDGETS[low_type].get(r, 0))

# =====
# fixed vocab for BIO entity tags
# -- only need single-direction transforms
class ExternalEntityVocab:
    ENTITY_TAGS = ["FAC", "GPE", "LOC", "ORG", "PER", "TTL", "VEH", "WEA"]
    ENTITY_FULL_LIST = ["O"] + [f"{p}-{t}" for t in ENTITY_TAGS for p in "BI"]
    ENTITY_MAP = {k:i for i,k in enumerate(ENTITY_FULL_LIST)}

    @staticmethod
    def idx_seq(tags):
        m = ExternalEntityVocab.ENTITY_MAP
        return [m.get(z, 0) for z in tags]  # 0 is the default "O"
