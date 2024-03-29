#

# onto for rams

# rams
ONTO_RAMS = {
'frames': [
{'name': 'artifactexistence.damagedestroy.damage', 'vp': 'damage', 'core_roles': ['artifact', 'damager', 'instrument', 'place'], 'template': '<damager> damage <artifact> with <instrument> in <place>'},
{'name': 'artifactexistence.damagedestroy.destroy', 'vp': 'destroy', 'core_roles': ['artifact', 'destroyer', 'instrument', 'place'], 'template': '<destroyer> destroy <artifact> with <instrument> in <place>'},
{'name': 'artifactexistence.damagedestroy.n/a', 'vp': 'damage or destroy', 'core_roles': ['artifact', 'damagerdestroyer', 'instrument', 'place'], 'template': '<damagerdestroyer> damage or destroy <artifact> with <instrument> in <place>'},
{'name': 'conflict.attack.airstrikemissilestrike', 'vp': 'strike', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> strike <target> with <instrument> in <place>'},
{'name': 'conflict.attack.biologicalchemicalpoisonattack', 'vp': 'attack', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> attack <target> with <instrument> in <place>'},
{'name': 'conflict.attack.bombing', 'vp': 'bomb', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> bomb <target> with <instrument> in <place>'},
{'name': 'conflict.attack.firearmattack', 'vp': 'attack', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> attack <target> with <instrument> in <place>'},
{'name': 'conflict.attack.hanging', 'vp': 'hang', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> hang <target> with <instrument> in <place>'},
{'name': 'conflict.attack.invade', 'vp': 'invade', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> invade <target> with <instrument> in <place>'},
{'name': 'conflict.attack.n/a', 'vp': 'attack', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> attack <target> with <instrument> in <place>'},
{'name': 'conflict.attack.selfdirectedbattle', 'vp': 'battle', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> battle against <target> with <instrument> in <place>'},
{'name': 'conflict.attack.setfire', 'vp': 'set fire', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> set fire on <target> with <instrument> in <place>'},
{'name': 'conflict.attack.stabbing', 'vp': 'stab', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> stab <target> with <instrument> in <place>'},
{'name': 'conflict.attack.stealrobhijack', 'vp': 'steal or rob', 'core_roles': ['artifact', 'attacker', 'instrument', 'place', 'target'], 'template': '<attacker> steal or rob <target> of <artifact> with <instrument> in <place> '},
{'name': 'conflict.attack.strangling', 'vp': 'strangle', 'core_roles': ['attacker', 'instrument', 'place', 'target'], 'template': '<attacker> strangle <target> with <instrument> in <place>'},
{'name': 'conflict.demonstrate.marchprotestpoliticalgathering', 'vp': 'demonstrate', 'core_roles': ['demonstrator', 'place'], 'template': '<demonstrator> demonstrate in <place>'},
{'name': 'conflict.demonstrate.n/a', 'vp': 'demonstrate', 'core_roles': ['demonstrator', 'place'], 'template': '<demonstrator> demonstrate in <place>'},
{'name': 'conflict.yield.n/a', 'vp': 'yield', 'core_roles': ['place', 'recipient', 'yielder'], 'template': '<yielder> yield to <recipient> in <place>'},
{'name': 'conflict.yield.retreat', 'vp': 'retreat', 'core_roles': ['destination', 'origin', 'retreater'], 'template': '<retreater> retreat from <origin> to <destination>'},
{'name': 'conflict.yield.surrender', 'vp': 'surrender', 'core_roles': ['place', 'recipient', 'surrenderer'], 'template': '<surrenderer> surrender to <recipient> in <place>'},
{'name': 'contact.collaborate.correspondence', 'vp': 'collaborate', 'core_roles': ['participant', 'place'], 'template': '<participant> collaborate in <place>'},
{'name': 'contact.collaborate.meet', 'vp': 'collaborate', 'core_roles': ['participant', 'place'], 'template': '<participant> collaborate in <place>'},
{'name': 'contact.collaborate.n/a', 'vp': 'collaborate', 'core_roles': ['participant', 'place'], 'template': '<participant> collaborate in <place>'},
{'name': 'contact.commandorder.broadcast', 'vp': 'order', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> order <recipient> in <place>'},
{'name': 'contact.commandorder.correspondence', 'vp': 'order', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> order <recipient> in <place>'},
{'name': 'contact.commandorder.meet', 'vp': 'order', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> order <recipient> in <place>'},
{'name': 'contact.commandorder.n/a', 'vp': 'order', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> order <recipient> in <place>'},
{'name': 'contact.commitmentpromiseexpressintent.broadcast', 'vp': 'promise', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> promise <recipient> in <place>'},
{'name': 'contact.commitmentpromiseexpressintent.correspondence', 'vp': 'promise', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> promise <recipient> in <place>'},
{'name': 'contact.commitmentpromiseexpressintent.meet', 'vp': 'promise', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> promise <recipient> in <place>'},
{'name': 'contact.commitmentpromiseexpressintent.n/a', 'vp': 'promise', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> promise <recipient> in <place>'},
{'name': 'contact.discussion.correspondence', 'vp': 'discuss', 'core_roles': ['participant', 'place'], 'template': '<participant> discuss in <place>'},
{'name': 'contact.discussion.meet', 'vp': 'discuss', 'core_roles': ['participant', 'place'], 'template': '<participant> discuss in <place>'},
{'name': 'contact.discussion.n/a', 'vp': 'discuss', 'core_roles': ['participant', 'place'], 'template': '<participant> discuss in <place>'},
{'name': 'contact.funeralvigil.meet', 'vp': 'attend the funeral', 'core_roles': ['deceased', 'participant', 'place'], 'template': '<participant> attend the funeral of <deceased> in <place>'},
{'name': 'contact.funeralvigil.n/a', 'vp': 'attend the funeral', 'core_roles': ['deceased', 'participant', 'place'], 'template': '<participant> attend the funeral of <deceased> in <place>'},
{'name': 'contact.mediastatement.broadcast', 'vp': 'state', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> state to <recipient> in <place>'},
{'name': 'contact.mediastatement.n/a', 'vp': 'state', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> state to <recipient> in <place>'},
{'name': 'contact.negotiate.correspondence', 'vp': 'negotiate', 'core_roles': ['participant', 'place'], 'template': '<participant> negotiate in <place>'},
{'name': 'contact.negotiate.meet', 'vp': 'negotiate', 'core_roles': ['participant', 'place'], 'template': '<participant> negotiate in <place>'},
{'name': 'contact.negotiate.n/a', 'vp': 'negotiate', 'core_roles': ['participant', 'place'], 'template': '<participant> negotiate in <place>'},
{'name': 'contact.prevarication.broadcast', 'vp': 'lie', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> lie to <recipient> in <place>'},
{'name': 'contact.prevarication.correspondence', 'vp': 'lie', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> lie to <recipient> in <place>'},
{'name': 'contact.prevarication.meet', 'vp': 'lie', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> lie to <recipient> in <place>'},
{'name': 'contact.prevarication.n/a', 'vp': 'lie', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> lie to <recipient> in <place>'},
{'name': 'contact.publicstatementinperson.broadcast', 'vp': 'state publicly', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> state publicly to <recipient> in <place>'},
{'name': 'contact.publicstatementinperson.n/a', 'vp': 'state publicly', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> state publicly to <recipient> in <place>'},
{'name': 'contact.requestadvise.broadcast', 'vp': 'request', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> request <recipient> in <place>'},
{'name': 'contact.requestadvise.correspondence', 'vp': 'request', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> request <recipient> in <place>'},
{'name': 'contact.requestadvise.meet', 'vp': 'request', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> request <recipient> in <place>'},
{'name': 'contact.requestadvise.n/a', 'vp': 'request', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> request <recipient> in <place>'},
{'name': 'contact.threatencoerce.broadcast', 'vp': 'threaten', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> threaten <recipient> in <place>'},
{'name': 'contact.threatencoerce.correspondence', 'vp': 'threaten', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> threaten <recipient> in <place>'},
{'name': 'contact.threatencoerce.meet', 'vp': 'threaten', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> threaten <recipient> in <place>'},
{'name': 'contact.threatencoerce.n/a', 'vp': 'threaten', 'core_roles': ['communicator', 'place', 'recipient'], 'template': '<communicator> threaten <recipient> in <place>'},
{'name': 'disaster.accidentcrash.accidentcrash', 'vp': 'crash', 'core_roles': ['crashobject', 'driverpassenger', 'place', 'vehicle'], 'template': '<driverpassenger> in <vehicle> crash with <crashobject> in <place> '},
{'name': 'disaster.fireexplosion.fireexplosion', 'vp': 'explode', 'core_roles': ['fireexplosionobject', 'instrument', 'place'], 'template': '<fireexplosionobject> explode with <instrument> in <place>'},
{'name': 'government.agreements.acceptagreementcontractceasefire', 'vp': 'sign agreement', 'core_roles': ['participant', 'place'], 'template': '<participant> sign agreement in <place>'},
{'name': 'government.agreements.n/a', 'vp': 'agree', 'core_roles': ['participant', 'place'], 'template': '<participant> agree in <place>'},
{'name': 'government.agreements.rejectnullifyagreementcontractceasefire', 'vp': 'reject agreement', 'core_roles': ['otherparticipant', 'place', 'rejecternullifier'], 'template': '<rejecternullifier> reject agreement with <otherparticipant> in <place>'},
{'name': 'government.agreements.violateagreement', 'vp': 'violate agreement', 'core_roles': ['otherparticipant', 'place', 'violator'], 'template': '<violator> violate agreement with <otherparticipant> in <place>'},
{'name': 'government.formation.mergegpe', 'vp': 'merge', 'core_roles': ['participant', 'place'], 'template': '<participant> merge in <place>'},
{'name': 'government.formation.n/a', 'vp': 'form', 'core_roles': ['founder', 'gpe', 'place'], 'template': '<founder> form <gpe> in <place>'},
{'name': 'government.formation.startgpe', 'vp': 'start', 'core_roles': ['founder', 'gpe', 'place'], 'template': '<founder> start <gpe> in <place>'},
{'name': 'government.legislate.legislate', 'vp': 'legislate', 'core_roles': ['governmentbody', 'law', 'place'], 'template': '<governmentbody> legislate <law> in <place>'},
{'name': 'government.spy.spy', 'vp': 'spy', 'core_roles': ['beneficiary', 'observedentity', 'place', 'spy'], 'template': '<spy> spy on <observedentity> for <beneficiary> in <place> '},
{'name': 'government.vote.castvote', 'vp': 'vote', 'core_roles': ['ballot', 'candidate', 'place', 'result', 'voter'], 'template': '<voter> vote for <candidate> on <ballot> with <result> in <place>'},
{'name': 'government.vote.n/a', 'vp': 'vote', 'core_roles': ['ballot', 'candidate', 'place', 'voter'], 'template': '<voter> vote for <candidate> on <ballot> in <place>'},
{'name': 'government.vote.violationspreventvote', 'vp': 'prevent', 'core_roles': ['ballot', 'candidate', 'place', 'preventer', 'voter'], 'template': '<preventer> prevent <voter> from voting for <candidate> on <ballot> in <place>'},
{'name': 'inspection.sensoryobserve.inspectpeopleorganization', 'vp': 'inspect', 'core_roles': ['inspectedentity', 'inspector', 'place'], 'template': '<inspector> inspect <inspectedentity> in <place>'},
{'name': 'inspection.sensoryobserve.monitorelection', 'vp': 'monitor', 'core_roles': ['monitor', 'monitoredentity', 'place'], 'template': '<monitor> monitor <monitoredentity> in <place>'},
{'name': 'inspection.sensoryobserve.n/a', 'vp': 'observe', 'core_roles': ['observedentity', 'observer', 'place'], 'template': '<observer> observe <observedentity> in <place>'},
{'name': 'inspection.sensoryobserve.physicalinvestigateinspect', 'vp': 'investigate', 'core_roles': ['inspectedentity', 'inspector', 'place'], 'template': '<inspector> investigate <inspectedentity> in <place>'},
{'name': 'justice.arrestjaildetain.arrestjaildetain', 'vp': 'arrest or jail', 'core_roles': ['crime', 'detainee', 'jailer', 'place'], 'template': '<jailer> arrest <detainee> for <crime> in <place>'},
{'name': 'justice.initiatejudicialprocess.chargeindict', 'vp': 'charge or indict', 'core_roles': ['crime', 'defendant', 'judgecourt', 'place', 'prosecutor'], 'template': '<prosecutor> charge or indict <defendant> for <crime> before <judgecourt> in <place> '},
{'name': 'justice.initiatejudicialprocess.n/a', 'vp': 'initiate judicial process', 'core_roles': ['crime', 'defendant', 'judgecourt', 'place', 'prosecutor'], 'template': '<prosecutor> initiate judicial process against <defendant> for <crime> before <judgecourt> in <place>'},
{'name': 'justice.initiatejudicialprocess.trialhearing', 'vp': 'trial or hearing', 'core_roles': ['crime', 'defendant', 'judgecourt', 'place', 'prosecutor'], 'template': '<prosecutor> try <defendant> for <crime> before <judgecourt> in <place>'},
{'name': 'justice.investigate.investigatecrime', 'vp': 'investigate', 'core_roles': ['crime', 'defendant', 'investigator', 'place'], 'template': '<investigator> investigate <defendant> for <crime> in <place>'},
{'name': 'justice.investigate.n/a', 'vp': 'investigate', 'core_roles': ['defendant', 'investigator', 'place'], 'template': '<investigator> investigate <defendant> in <place>'},
{'name': 'justice.judicialconsequences.convict', 'vp': 'convict', 'core_roles': ['crime', 'defendant', 'judgecourt', 'place'], 'template': '<judgecourt> convict <defendant> for <crime> in <place>'},
{'name': 'justice.judicialconsequences.execute', 'vp': 'execute', 'core_roles': ['crime', 'defendant', 'executioner', 'place'], 'template': '<executioner> execute <defendant> for <crime> in <place>'},
{'name': 'justice.judicialconsequences.extradite', 'vp': 'extradite', 'core_roles': ['crime', 'defendant', 'destination', 'extraditer', 'origin'], 'template': '<extraditer> extradite <defendant> for <crime> from <destination> to <origin>'},
{'name': 'justice.judicialconsequences.n/a', 'vp': 'decide consequence', 'core_roles': ['crime', 'defendant', 'judgecourt', 'place'], 'template': '<judgecourt> decide consequence against <defendant> for <crime> in <place>'},
{'name': 'life.die.deathcausedbyviolentevents', 'vp': 'kill', 'core_roles': ['instrument', 'killer', 'place', 'victim'], 'template': '<killer> kill <victim> with <instrument> in <place>'},
{'name': 'life.die.n/a', 'vp': 'die', 'core_roles': ['place', 'victim'], 'template': '<victim> die in <place>'},
{'name': 'life.die.nonviolentdeath', 'vp': 'die', 'core_roles': ['place', 'victim'], 'template': '<victim> die in <place>'},
{'name': 'life.injure.illnessdegradationhungerthirst', 'vp': 'experience hunger or thirst', 'core_roles': ['place', 'victim'], 'template': '<victim> experience hunger or thirst in <place>'},
{'name': 'life.injure.illnessdegradationphysical', 'vp': 'degrade physically', 'core_roles': ['victim', 'place'], 'template': '<victim> degrade physically in <place>'},
{'name': 'life.injure.injurycausedbyviolentevents', 'vp': 'injure', 'core_roles': ['injurer', 'instrument', 'place', 'victim'], 'template': '<injurer> injure <victim> with <instrument> in <place>'},
{'name': 'life.injure.n/a', 'vp': 'injure', 'core_roles': ['injurer', 'place', 'victim'], 'template': '<injurer> injure <victim> in <place>'},
{'name': 'manufacture.artifact.build', 'vp': 'build', 'core_roles': ['artifact', 'instrument', 'manufacturer', 'place'], 'template': '<manufacturer> build <artifact> with <instrument> in <place>'},
{'name': 'manufacture.artifact.createintellectualproperty', 'vp': 'create', 'core_roles': ['artifact', 'instrument', 'manufacturer', 'place'], 'template': '<manufacturer> create <artifact> with <instrument> in <place>'},
{'name': 'manufacture.artifact.createmanufacture', 'vp': 'create', 'core_roles': ['artifact', 'instrument', 'manufacturer', 'place'], 'template': '<manufacturer> create <artifact> with <instrument> in <place>'},
{'name': 'manufacture.artifact.n/a', 'vp': 'manufacture', 'core_roles': ['artifact', 'instrument', 'manufacturer', 'place'], 'template': '<manufacturer> manufacture <artifact> with <instrument> in <place>'},
{'name': 'movement.transportartifact.bringcarryunload', 'vp': 'carry', 'core_roles': ['artifact', 'destination', 'origin', 'transporter', 'vehicle'], 'template': '<transporter> carry <artifact> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportartifact.disperseseparate', 'vp': 'disperse', 'core_roles': ['artifact', 'destination', 'origin', 'transporter', 'vehicle'], 'template': '<transporter> disperse <artifact> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportartifact.fall', 'vp': 'fall', 'core_roles': ['artifact', 'destination', 'origin'], 'template': '<artifact> fall <origin> from to <destination>'},
{'name': 'movement.transportartifact.grantentry', 'vp': 'grant entry', 'core_roles': ['artifact', 'destination', 'origin', 'transporter'], 'template': '<transporter> grant entry of <artifact> from <origin> to <destination>'},
{'name': 'movement.transportartifact.hide', 'vp': 'hide', 'core_roles': ['artifact', 'hidingplace', 'origin', 'transporter', 'vehicle'], 'template': '<transporter> hide <artifact> inside <vehicle> from <origin> in <hidingplace>'},
{'name': 'movement.transportartifact.n/a', 'vp': 'transport', 'core_roles': ['artifact', 'destination', 'origin', 'transporter', 'vehicle'], 'template': '<transporter> transport <artifact> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportartifact.nonviolentthrowlaunch', 'vp': 'throw', 'core_roles': ['artifact', 'destination', 'origin', 'transporter', 'vehicle'], 'template': '<transporter> throw <artifact> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportartifact.prevententry', 'vp': 'prevent entry', 'core_roles': ['artifact', 'destination', 'origin', 'preventer', 'transporter'], 'template': '<preventer> prevent <transporter> from transporting <artifact> from <origin> to <destination>'},
{'name': 'movement.transportartifact.preventexit', 'vp': 'prevent exit', 'core_roles': ['artifact', 'destination', 'origin', 'preventer', 'transporter'], 'template': '<preventer> prevent <transporter> from transporting <artifact> from <origin> to <destination>'},
{'name': 'movement.transportartifact.receiveimport', 'vp': 'import', 'core_roles': ['artifact', 'destination', 'origin', 'transporter', 'vehicle'], 'template': '<transporter> import <artifact> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportartifact.sendsupplyexport', 'vp': 'export', 'core_roles': ['artifact', 'destination', 'origin', 'transporter', 'vehicle'], 'template': '<transporter> export <artifact> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportartifact.smuggleextract', 'vp': 'smuggleextract', 'core_roles': ['artifact', 'destination', 'origin', 'transporter', 'vehicle'], 'template': '<transporter> smuggle <artifact> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportperson.bringcarryunload', 'vp': 'carry', 'core_roles': ['destination', 'origin', 'passenger', 'transporter', 'vehicle'], 'template': '<transporter> carry <passenger> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportperson.disperseseparate', 'vp': 'disperse', 'core_roles': ['destination', 'origin', 'passenger', 'transporter', 'vehicle'], 'template': '<transporter> disperse <passenger> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportperson.evacuationrescue', 'vp': 'rescue', 'core_roles': ['destination', 'origin', 'passenger', 'transporter', 'vehicle'], 'template': '<transporter> rescue <passenger> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportperson.fall', 'vp': 'fall', 'core_roles': ['destination', 'origin', 'passenger'], 'template': '<passenger> fall <origin> from to <destination>'},
{'name': 'movement.transportperson.grantentryasylum', 'vp': 'grantentryasylum', 'core_roles': ['destination', 'granter', 'origin', 'passenger', 'transporter'], 'template': '<granter> grant entry to <transporter> transporting <passenger> from <origin> to <destination>'},
{'name': 'movement.transportperson.hide', 'vp': 'hide', 'core_roles': ['hidingplace', 'origin', 'passenger', 'transporter', 'vehicle'], 'template': '<transporter> hide <passenger> inside <vehicle> from <origin> in <hidingplace>'},
{'name': 'movement.transportperson.n/a', 'vp': 'transport', 'core_roles': ['destination', 'origin', 'passenger', 'transporter', 'vehicle'], 'template': '<transporter> transport <passenger> on <vehicle> from <origin> to <destination>'},
{'name': 'movement.transportperson.prevententry', 'vp': 'prevent entry', 'core_roles': ['destination', 'origin', 'passenger', 'preventer', 'transporter'], 'template': '<preventer> prevent <transporter> from transporting <passenger> from <origin> to <destination>'},
{'name': 'movement.transportperson.preventexit', 'vp': 'prevent exit', 'core_roles': ['destination', 'origin', 'passenger', 'preventer', 'transporter'], 'template': '<preventer> prevent <transporter> from transporting <passenger> from <origin> to <destination>'},
{'name': 'movement.transportperson.selfmotion', 'vp': 'move', 'core_roles': ['destination', 'origin', 'transporter'], 'template': '<transporter> move from <origin> to <destination>'},
{'name': 'movement.transportperson.smuggleextract', 'vp': 'smuggle', 'core_roles': ['destination', 'origin', 'passenger', 'transporter', 'vehicle'], 'template': '<transporter> smuggle <passenger> on <vehicle> from <origin> to <destination>'},
{'name': 'personnel.elect.n/a', 'vp': 'elect', 'core_roles': ['candidate', 'place', 'voter'], 'template': '<voter> elect <candidate> in <place>'},
{'name': 'personnel.elect.winelection', 'vp': 'elect', 'core_roles': ['candidate', 'place', 'voter'], 'template': '<voter> elect <candidate> in <place>'},
{'name': 'personnel.endposition.firinglayoff', 'vp': 'stop working', 'core_roles': ['employee', 'place', 'placeofemployment'], 'template': '<employee> stop working for <placeofemployment> in <place>'},
{'name': 'personnel.endposition.n/a', 'vp': 'stop working', 'core_roles': ['employee', 'place', 'placeofemployment'], 'template': '<employee> stop working for <placeofemployment> in <place>'},
{'name': 'personnel.endposition.quitretire', 'vp': 'quit or retire', 'core_roles': ['employee', 'place', 'placeofemployment'], 'template': '<employee> quit or retire from <placeofemployment> in <place>'},
{'name': 'personnel.startposition.hiring', 'vp': 'hire', 'core_roles': ['employee', 'place', 'placeofemployment'], 'template': '<placeofemployment> hire <employee> in <place>'},
{'name': 'personnel.startposition.n/a', 'vp': 'start working', 'core_roles': ['employee', 'place', 'placeofemployment'], 'template': '<employee> start working for <placeofemployment> in <place>'},
{'name': 'transaction.transaction.embargosanction', 'vp': 'prevent', 'core_roles': ['artifactmoney', 'giver', 'place', 'preventer', 'recipient'], 'template': '<preventer> prevent <giver> from giving <artifactmoney> to <recipient> in <place>'},
{'name': 'transaction.transaction.giftgrantprovideaid', 'vp': 'give aid', 'core_roles': ['beneficiary', 'giver', 'place', 'recipient'], 'template': '<giver> give aid to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transaction.n/a', 'vp': 'transact', 'core_roles': ['beneficiary', 'participant', 'place'], 'template': '<participant> transact for <beneficiary> in <place>'},
{'name': 'transaction.transaction.transfercontrol', 'vp': 'transfer', 'core_roles': ['beneficiary', 'giver', 'place', 'recipient', 'territoryorfacility'], 'template': '<giver> transfer <territoryorfacility> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transfermoney.borrowlend', 'vp': 'lend', 'core_roles': ['beneficiary', 'giver', 'money', 'place', 'recipient'], 'template': '<giver> lend <money> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transfermoney.embargosanction', 'vp': 'prevent', 'core_roles': ['giver', 'money', 'place', 'preventer', 'recipient'], 'template': '<preventer> prevent <giver> from giving <money> to <recipient> in <place>'},
{'name': 'transaction.transfermoney.giftgrantprovideaid', 'vp': 'give', 'core_roles': ['beneficiary', 'giver', 'money', 'place', 'recipient'], 'template': '<giver> give <money> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transfermoney.n/a', 'vp': 'transfer', 'core_roles': ['beneficiary', 'giver', 'money', 'place', 'recipient'], 'template': '<giver> give <money> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transfermoney.payforservice', 'vp': 'pay', 'core_roles': ['beneficiary', 'giver', 'money', 'place', 'recipient'], 'template': '<giver> pay <money> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transfermoney.purchase', 'vp': 'give', 'core_roles': ['beneficiary', 'giver', 'money', 'place', 'recipient'], 'template': '<giver> pay <money> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transferownership.borrowlend', 'vp': 'lend', 'core_roles': ['artifact', 'beneficiary', 'giver', 'place', 'recipient'], 'template': '<giver> lend <artifact> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transferownership.embargosanction', 'vp': 'prevent', 'core_roles': ['artifact', 'giver', 'place', 'preventer', 'recipient'], 'template': '<preventer> prevent <giver> from giving <artifact> to <recipient> in <place>'},
{'name': 'transaction.transferownership.giftgrantprovideaid', 'vp': 'give', 'core_roles': ['artifact', 'beneficiary', 'giver', 'place', 'recipient'], 'template': '<giver> give <artifact> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transferownership.n/a', 'vp': 'give', 'core_roles': ['artifact', 'beneficiary', 'giver', 'place', 'recipient'], 'template': '<giver> give <artifact> to <recipient> for <beneficiary> in <place>'},
{'name': 'transaction.transferownership.purchase', 'vp': 'purchase', 'core_roles': ['artifact', 'beneficiary', 'giver', 'place', 'recipient'], 'template': '<giver> purchase <artifact> from <recipient> for <beneficiary> in <place>'},
],
'roles': [
{'name': 'artifact', 'np': 'artifact'},
{'name': 'artifactmoney', 'np': 'artifact or money'},
{'name': 'attacker', 'np': 'attacker', 'qwords': ['who']},
{'name': 'ballot', 'np': 'ballot'},
{'name': 'beneficiary', 'np': 'beneficiary', 'qwords': ['who']},
{'name': 'candidate', 'np': 'candidate', 'qwords': ['who']},
{'name': 'communicator', 'np': 'communicator', 'qwords': ['who']},
{'name': 'crashobject', 'np': 'crashed object'},
{'name': 'crime', 'np': 'crime'},
{'name': 'damager', 'np': 'agent', 'qwords': ['who']},
{'name': 'damagerdestroyer', 'np': 'agent', 'qwords': ['who']},
{'name': 'deceased', 'np': 'deceased person', 'qwords': ['who']},
{'name': 'defendant', 'np': 'defendant', 'qwords': ['who']},
{'name': 'demonstrator', 'np': 'demonstrator', 'qwords': ['who']},
{'name': 'destination', 'np': 'destination', 'qwords': ['where2']},
{'name': 'destroyer', 'np': 'agent', 'qwords': ['who']},
{'name': 'detainee', 'np': 'detainee', 'qwords': ['who']},
{'name': 'driverpassenger', 'np': 'driver', 'qwords': ['who']},
{'name': 'employee', 'np': 'employee', 'qwords': ['who']},
{'name': 'executioner', 'np': 'agent', 'qwords': ['who']},
{'name': 'extraditer', 'np': 'agent', 'qwords': ['who']},
{'name': 'fireexplosionobject', 'np': 'explosive object'},
{'name': 'founder', 'np': 'founder', 'qwords': ['who']},
{'name': 'giver', 'np': 'giver', 'qwords': ['who']},
{'name': 'governmentbody', 'np': 'government'},
{'name': 'gpe', 'np': 'entity'},
{'name': 'granter', 'np': 'granter', 'qwords': ['who']},
{'name': 'hidingplace', 'np': 'hiding place', 'qwords': ['where2']},
{'name': 'injurer', 'np': 'injurer', 'qwords': ['who']},
{'name': 'inspectedentity', 'np': 'entity'},
{'name': 'inspector', 'np': 'inspector', 'qwords': ['who']},
{'name': 'instrument', 'np': 'instrument'},
{'name': 'investigator', 'np': 'investigator', 'qwords': ['who']},
{'name': 'jailer', 'np': 'jailer', 'qwords': ['who']},
{'name': 'judgecourt', 'np': 'judge or court'},
{'name': 'killer', 'np': 'killer', 'qwords': ['who']},
{'name': 'law', 'np': 'law'},
{'name': 'manufacturer', 'np': 'manufacturer', 'qwords': ['who']},
{'name': 'money', 'np': 'money'},
{'name': 'monitor', 'np': 'monitor', 'qwords': ['who']},
{'name': 'monitoredentity', 'np': 'entity'},
{'name': 'observedentity', 'np': 'entity'},
{'name': 'observer', 'np': 'observer', 'qwords': ['who']},
{'name': 'origin', 'np': 'origin', 'qwords': ['where2']},
{'name': 'otherparticipant', 'np': 'other participant', 'qwords': ['who']},
{'name': 'participant', 'np': 'participant', 'qwords': ['who']},
{'name': 'passenger', 'np': 'passenger', 'qwords': ['who']},
{'name': 'place', 'np': 'place', 'qwords': ['where', 'where2']},
{'name': 'placeofemployment', 'np': 'employer'},
{'name': 'preventer', 'np': 'agent', 'qwords': ['who']},
{'name': 'prosecutor', 'np': 'prosecutor', 'qwords': ['who']},
{'name': 'recipient', 'np': 'recipient', 'qwords': ['who']},
{'name': 'rejecternullifier', 'np': 'agent', 'qwords': ['who']},
{'name': 'result', 'np': 'result'},
{'name': 'retreater', 'np': 'agent', 'qwords': ['who']},
{'name': 'spy', 'np': 'spy', 'qwords': ['who']},
{'name': 'surrenderer', 'np': 'agent', 'qwords': ['who']},
{'name': 'target', 'np': 'target'},
{'name': 'territoryorfacility', 'np': 'territory or facility'},
{'name': 'transporter', 'np': 'transporter', 'qwords': ['who']},
{'name': 'vehicle', 'np': 'vehicle'},
{'name': 'victim', 'np': 'victim', 'qwords': ['who']},
{'name': 'violator', 'np': 'violator', 'qwords': ['who']},
{'name': 'voter', 'np': 'voter', 'qwords': ['who']},
{'name': 'yielder', 'np': 'agent', 'qwords': ['who']},
],
}

# --
# bionlp11 (borrow place in this file ...)
ONTO_BIO11 = {
'frames': [
{'name': 'Gene_expression', 'vp': 'express', 'core_roles': ['Agent', 'Theme'], 'template': '<Agent> express <Theme>',
 'role_questions': {'Agent': 'What?', 'Theme': 'What is expressed?'}},
{'name': 'Transcription', 'vp': 'transcribe', 'core_roles': ['Agent', 'Theme'], 'template': '<Agent> transcribe <Theme>',
 'role_questions': {'Agent': 'What?', 'Theme': 'What is transcribed?'}},
{'name': 'Protein_catabolism', 'vp': 'degrade', 'core_roles': ['Agent', 'Theme'], 'template': '<Agent> degrade <Theme>',
 'role_questions': {'Agent': 'What?', 'Theme': 'What is degraded?'}},
{'name': 'Phosphorylation', 'vp': 'phosphorylate', 'core_roles': ['Agent', 'Theme'], 'template': '<Agent> phosphorylate <Theme>',
 'role_questions': {'Agent': 'What?', 'Theme': 'What is phosphorylated?'}},
{'name': 'Localization', 'vp': 'localize', 'core_roles': ['Agent', 'Theme'], 'template': '<Agent> localize <Theme>',
 'role_questions': {'Agent': 'What?', 'Theme': 'What is localized?'}},
{'name': 'Binding', 'vp': 'bind', 'core_roles': ['Agent', 'Theme1', 'Theme2'], 'template': '<Agent> bind <Theme1> to <Theme2>',
 'role_questions': {'Agent': 'What?', 'Theme1': 'What is bound?', 'Theme2': 'What is something bound to?'}},
{'name': 'Regulation', 'vp': 'regulate', 'core_roles': ['Cause', 'Theme'], 'template': '<Cause> regulate <Theme>',
 'role_questions': {'Theme': 'What is regulated?', 'Cause': 'What causes the regulation?'}},
{'name': 'Positive_regulation', 'vp': 'regulate', 'core_roles': ['Cause', 'Theme'], 'template': '<Cause> regulate <Theme>',
 'role_questions': {'Theme': 'What is regulated?', 'Cause': 'What causes the regulation?'}},
{'name': 'Negative_regulation', 'vp': 'regulate', 'core_roles': ['Cause', 'Theme'], 'template': '<Cause> regulate <Theme>',
 'role_questions': {'Theme': 'What is regulated?', 'Cause': 'What causes the regulation?'}},
],
'roles': [
{'name': 'Agent', 'np': 'agent'},
{'name': 'Cause', 'np': 'cause'},
{'name': 'Theme', 'np': 'theme'},
{'name': 'Theme1', 'np': 'theme 1'},
{'name': 'Theme2', 'np': 'theme 2'},
],
}
# --
