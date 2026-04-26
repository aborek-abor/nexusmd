"""
NexusMD — Built-in compound library for scaffold hopping.
Includes African phytochemicals, TCM compounds, and key FDA drugs.
"""

LIBRARY_SMILES = [
    # African Phytochemicals
    {"name":"Artemisinin","smiles":"O=C1OC2CC3(C)CCCC(=O)C3(C)C2O1","source":"African","mw":282,"logp":0.2},
    {"name":"Berberine","smiles":"COc1ccc2CC3=CC=CC=C3[N+]2=Cc2cc(OC)c(OC)cc21","source":"African","mw":336,"logp":1.3},
    {"name":"Quercetin","smiles":"O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12","source":"African","mw":302,"logp":1.5},
    {"name":"Plumbagin","smiles":"Cc1cc(=O)c2cccc(O)c2c1=O","source":"African","mw":188,"logp":2.1},
    {"name":"Cryptolepine","smiles":"c1ccc2nc3ccccc3cc2c1","source":"African","mw":262,"logp":1.7},
    {"name":"Luteolin","smiles":"O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12","source":"African","mw":286,"logp":2.4},
    {"name":"Harmine","smiles":"COc1ccc2[nH]c3ccccc3c2c1","source":"African","mw":212,"logp":1.6},
    {"name":"Thymoquinone","smiles":"O=C1C(=O)C(C(C)C)=CC(C)=C1","source":"African","mw":164,"logp":1.8},
    {"name":"Resveratrol","smiles":"Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1","source":"African","mw":228,"logp":3.1},
    {"name":"Kaempferol","smiles":"O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12","source":"African","mw":286,"logp":1.9},
    {"name":"Camptothecin","smiles":"O=C1OCC(=O)c2cc3ccccc3nc21","source":"African","mw":348,"logp":1.1},
    {"name":"Piperine","smiles":"O=C(/C=C/C=C/c1ccc2c(c1)OCO2)N1CCCCC1","source":"African","mw":285,"logp":2.6},
    {"name":"Eugenol","smiles":"C=CCc1ccc(O)c(OC)c1","source":"African","mw":164,"logp":2.7},
    {"name":"Physostigmine","smiles":"CNC(=O)Oc1ccc2c(c1)[C@@H]1CC[C@]2(C)N1C","source":"African","mw":275,"logp":1.12},
    {"name":"Apigenin","smiles":"O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12","source":"African","mw":270,"logp":2.8},
    # TCM Phytochemicals
    {"name":"Baicalein","smiles":"O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12","source":"TCM","mw":270,"logp":2.0},
    {"name":"Tanshinone IIA","smiles":"CC1(C)CCCC2=C1C=CC1=CC(=O)c3cccc(C)c3C12","source":"TCM","mw":294,"logp":3.0},
    {"name":"Andrographolide","smiles":"OCC(=O)[C@H]1CC[C@@H]2[C@@](C)(CO)CC[C@@H]2[C@H]1C","source":"TCM","mw":350,"logp":1.1},
    {"name":"Curcumin","smiles":"O=C(/C=C/c1ccc(O)c(OC)c1)CC(=O)/C=C/c1ccc(O)c(OC)c1","source":"TCM","mw":368,"logp":3.3},
    {"name":"Emodin","smiles":"Cc1cc(O)c2cc(O)cc(=O)c2c1O","source":"TCM","mw":270,"logp":2.3},
    {"name":"Matrine","smiles":"[C@@H]1(CC[C@H]2CCCN3CCC[C@@H]([C@@H]1O2)N3)=O","source":"TCM","mw":248,"logp":0.6},
    {"name":"Celastrol","smiles":"CC1(C)CCC2=CC(=O)c3cc(C(=O)O)ccc3C2=C1","source":"TCM","mw":450,"logp":5.8},
    {"name":"Honokiol","smiles":"C=CCc1ccc(Oc2ccc(C/C=C\c3ccccc3)cc2)cc1","source":"TCM","mw":266,"logp":3.8},
    {"name":"Wogonin","smiles":"COc1c(O)cc2oc(-c3ccccc3)cc(=O)c2c1O","source":"TCM","mw":284,"logp":2.9},
    {"name":"Naringenin","smiles":"O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21","source":"TCM","mw":272,"logp":2.1},
    # FDA Approved Drugs
    {"name":"Lopinavir","smiles":"CC1=CC=C(C=C1)S(=O)(=O)NC(CC(CC(NC(=O)C2N(CC3)CCN3C(=O)C(CC(C)C)NC(=O)COC4=CC=CC=C4C)Cc5ccccc5)O)C(C)C","source":"FDA","mw":628,"logp":5.0},
    {"name":"Ritonavir","smiles":"CC(C)c1csc(NC(=O)NC(Cc2ccccc2)C(O)CN(CC(C)C)C(=O)c2csc(C(C)C)n2)n1","source":"FDA","mw":721,"logp":5.0},
    {"name":"Darunavir","smiles":"CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC2COC3CCCC23)S(=O)(=O)c1ccc(N)cc1","source":"FDA","mw":547,"logp":2.4},
    {"name":"Imatinib","smiles":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1","source":"FDA","mw":493,"logp":3.7},
    {"name":"Gefitinib","smiles":"COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1","source":"FDA","mw":446,"logp":3.2},
    {"name":"Erlotinib","smiles":"C#Cc1cc2c(Nc3ccccc3OCC)ncnc2cc1OCCCOC","source":"FDA","mw":393,"logp":2.7},
    {"name":"Sunitinib","smiles":"CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\C(=O)Nc3cc(F)ccc32)c1C","source":"FDA","mw":398,"logp":3.0},
    {"name":"Sorafenib","smiles":"CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1","source":"FDA","mw":464,"logp":3.8},
    {"name":"Osimertinib","smiles":"C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1N(C)CCN1CCOCC1","source":"FDA","mw":499,"logp":3.5},
    {"name":"Venetoclax","smiles":"CC1(CCC(=C1)c1ccc(Cl)c(c1)-n1cc(-c2ccc3c(c2)C(=O)NCC3)nn1)CN1CCOCC1","source":"FDA","mw":868,"logp":6.4},
]
