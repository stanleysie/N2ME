import os
import json
import pandas as pd

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

# directories
samm_aus = handle_windows_path("path to SAMM extracted AUS")
mmew_aus = handle_windows_path("path to MMEW extracted AUS")
casme_ii_aus = handle_windows_path("path to CASME II extracted AUS")

# FACS files
samm_facs = handle_windows_path("path to SAMM FACS annotation (.csv)")
mmew_facs = handle_windows_path("path to MMEW FACS annotation (.csv)")
casme_ii_facs = handle_windows_path("path to CASME II FACS annotation (.csv)")

def prepare_me_summary(save_dir):
    samm_df = pd.read_csv(samm_facs)
    mmew_df = pd.read_csv(mmew_facs)
    casme_ii_df = pd.read_csv(casme_ii_facs)

    # formatting columns
    samm_df = samm_df[['Filename', 'Estimated Emotion']]
    mmew_df = mmew_df[['Filename', 'Estimated Emotion']]
    casme_ii_df['Subject'] = casme_ii_df['Subject'].apply(lambda x: f'{int(x):02d}')
    casme_ii_df['Filename'] = 'sub' + casme_ii_df['Subject'] + '_' + casme_ii_df['Filename']
    casme_ii_df = casme_ii_df[['Filename', 'Estimated Emotion']]

    expressions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

    samm_df = samm_df[samm_df['Estimated Emotion'].str.lower().isin(expressions)]
    mmew_df = mmew_df[mmew_df['Estimated Emotion'].str.lower().isin(expressions)]
    casme_ii_df = casme_ii_df[casme_ii_df['Estimated Emotion'].str.lower().isin(expressions)]
    samm_df = samm_df.to_dict('records')
    mmew_df = mmew_df.to_dict('records')
    casme_ii_df = casme_ii_df.to_dict('records')

    subject_emo = {}
    for expr in expressions:
        subject_emo[expr] = []

    for d in samm_df:
        subject_emo[d['Estimated Emotion'].lower()]
        subject_emo[d['Estimated Emotion'].lower()].append(f"s_{d['Filename']}")

    for d in mmew_df:
        subject_emo[d['Estimated Emotion'].lower()]
        subject_emo[d['Estimated Emotion'].lower()].append(f"m_{d['Filename']}")

    for d in casme_ii_df:
        subject_emo[d['Estimated Emotion'].lower()]
        subject_emo[d['Estimated Emotion'].lower()].append(f"c_{d['Filename']}")

    with open(os.path.join(save_dir, 'me_summary.json'), 'w') as j:
        json.dump(subject_emo, j, indent=4)