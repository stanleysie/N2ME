import glob
import random
import pandas as pd

class DataExtractor:
    def __init__(self, emotions, facs_dir):
        self.emotions = emotions
        self.facs_dir = facs_dir
        self.load_facs()
    

    def load_facs(self):
        samm_annotation = self.facs_dir['SAMM']
        samm_df = pd.read_csv(samm_annotation)
        samm_df.drop(labels=['Subject', 'Inducement Code',  'Micro', 'Objective Classes', 'Notes'], axis=1, inplace=True)
        samm_df.rename(columns={ 'Onset Frame': 'OnsetFrame', 'Apex Frame': 'ApexFrame', 'Offset Frame': 'OffsetFrame' }, inplace=True)
        samm_df['Subject'] = pd.Series([s.split('_')[0] for s in list(samm_df['Filename'])])
        samm_df = samm_df[samm_df['Estimated Emotion'].str.lower().isin(self.emotions)]
        samm_df = samm_df[['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Duration', 'Action Units', 'Estimated Emotion']]

        mmew_annotation = self.facs_dir['MMEW']
        mmew_df = pd.read_csv(mmew_annotation)
        mmew_df['Estimated Emotion'] = mmew_df['Estimated Emotion'].apply(lambda x: x.title())
        mmew_df['Duration'] = mmew_df['OffsetFrame'] - mmew_df['OnsetFrame'] + 1
        mmew_df.drop(labels=['remarks'], axis=1, inplace=True)
        mmew_df = mmew_df[mmew_df['Estimated Emotion'].str.lower().isin(self.emotions)]
        mmew_df = mmew_df[['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Duration', 'Action Units', 'Estimated Emotion']]

        casme_ii_annotation = self.facs_dir['CASME_II']
        casme_ii_df = pd.read_csv(casme_ii_annotation)
        casme_ii_df['Subject'] = casme_ii_df['Subject'].apply(lambda x: f'sub{x:02d}')
        casme_ii_df['Filename'] = casme_ii_df['Subject'] + '_' + casme_ii_df['Filename']
        casme_ii_df['Estimated Emotion'] = casme_ii_df['Estimated Emotion'].apply(lambda x: x.title())
        casme_ii_df['Duration'] = casme_ii_df['OffsetFrame'] - casme_ii_df['OnsetFrame'] + 1
        casme_ii_df = casme_ii_df[casme_ii_df['ApexFrame'] != '/']  # edge case
        casme_ii_df = casme_ii_df[casme_ii_df['Estimated Emotion'].str.lower().isin(self.emotions)]
        casme_ii_df = casme_ii_df[['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Duration', 'Action Units', 'Estimated Emotion']]

        self.data_df = {
            's': samm_df,
            'm': mmew_df,
            'c': casme_ii_df
        }
    

    def get_data(self, data, n_samples):
        data_dict = {
            'onset': [],
            'apex': []
        }
        taken_index = []

        count = 0
        while count < n_samples:
            index = random.randint(0, len(data)-1)
            sample = data[index]

            if index in taken_index:
                continue

            me_ref = sample.split('/')[-2]
            db = 'SAMM'
            if me_ref.startswith('m_'):
                db = 'MMEW'
            elif me_ref.startswith('c_'):
                db = 'CASME_II'

            # extract reference onset and apex from FACS annotation
            prefix = db[0].lower()
            subject = me_ref[2:]
            df = self.data_df[prefix]
            df = df[df['Filename'] == subject].to_dict('records')[0]
            onset = df['OnsetFrame']
            apex = df['ApexFrame']
            if onset == apex:
                continue

            diff = abs(0 - onset)
            onset -= diff
            apex -= diff

            me = glob.glob(f'{sample}/*.jpg')
            me.sort()
            onset = me[onset]
            apex = me[apex]

            data_dict['onset'].append((f"{prefix}_{df['Subject']}", onset))
            data_dict['apex'].append((f"{prefix}_{df['Subject']}", apex))
            
            count += 1
            taken_index.append(index)
            if count == n_samples:
                break

        data_dict['onset'] = list(set(data_dict['onset']))
        data_dict['apex'] = list(set(data_dict['apex']))
        data_dict['onset'] = sorted(data_dict['onset'], key=lambda x: x[1])
        data_dict['apex'] = sorted(data_dict['apex'], key=lambda x: x[1])
        
        return data_dict