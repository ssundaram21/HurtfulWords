import os
import sys
import subprocess
import shlex
import torch

cols = ['Acute and unspecified renal failure',
 'Acute cerebrovascular disease',
 'Acute myocardial infarction',
 'Cardiac dysrhythmias',
 'Chronic kidney disease',
 'Chronic obstructive pulmonary disease and bronchiectasis',
 'Complications of surgical procedures or medical care',
 'Conduction disorders',
 'Congestive heart failure; nonhypertensive',
 'Coronary atherosclerosis and other heart disease',
 'Diabetes mellitus with complications',
 'Diabetes mellitus without complication',
 'Disorders of lipid metabolism',
 'Essential hypertension',
 'Fluid and electrolyte disorders',
 'Gastrointestinal hemorrhage',
 'Hypertension with complications and secondary hypertension',
 'Other liver diseases',
 'Other lower respiratory disease',
 'Other upper respiratory disease',
 'Pleurisy; pneumothorax; pulmonary collapse',
 'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
 'Respiratory failure; insufficiency; arrest (adult)',
 'Septicemia (except in labor)',
 'Shock',
 'any_chronic',
 'any_acute',
 'any_disease']

std_models = ['baseline_clinical_BERT_1_epoch_512']#, 'adv_clinical_BERT_gender_1_epoch_512']

# file name, col names, models
# tasks = [('inhosp_mort', ['inhosp_mort'],  std_models),
#         ('phenotype_all', cols, std_models),
#          ('phenotype_first', cols, std_models) ]
tasks = [('inhosp_mort', ['inhosp_mort'],  std_models)]

print("Device: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

for dfname, targetnames, models in tasks:
    for t in targetnames:
        for c,m in enumerate(models):
            print(t)
            subprocess.call(shlex.split('sbatch finetune_on_target.sh "%s" "%s" "%s"'%(dfname,m,t)))

