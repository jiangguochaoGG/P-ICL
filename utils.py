import random

class Prompt:
    def __init__(self, dataset):
        self.dataset = dataset
    
        if self.dataset not in ['CoNLL2003', 'ACE2004', 'ACE2005', 'WNUT2017']:
            raise NotImplementedError(f'This dataset {self.dataset} is not used!')         

        if self.dataset == 'CoNLL2003':
            self.baseline_prompt = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- PER\n'
                '- ORG\n'
                '- LOC\n'
                '- MISC\n'
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.icl_prompt1 = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- PER\n'
                '- ORG\n'
                '- LOC\n'
                '- MISC\n'
                'Here are some examples:\n'
            )
            self.icl_prompt2 = (
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.picl_prompt1 = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- PER: e.g. {PER}\n'
                '- ORG: e.g. {ORG}\n'
                '- LOC: e.g. {LOC}\n'
                '- MISC: e.g. {MISC}\n'
            )
            self.picl_prompt2 = (
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.fusion_prompt = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- PER: e.g. {PER}\n'
                '- ORG: e.g. {ORG}\n'
                '- LOC: e.g. {LOC}\n'
                '- MISC: e.g. {MISC}\n'
                'Here are some examples:\n'
            )
        elif self.dataset == 'WNUT2017':
            self.baseline_prompt = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- person\n'
                '- location\n'
                '- corporation\n'
                '- product\n'
                '- creative-work\n'
                '- group\n'
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.icl_prompt1 = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- person\n'
                '- location\n'
                '- corporation\n'
                '- product\n'
                '- creative-work\n'
                '- group\n'
                'Here are some examples:\n'
            )
            self.icl_prompt2 = (
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.picl_prompt1 = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- person: e.g. {person}\n'
                '- location: e.g. {location}\n'
                '- corporation: e.g. {corporation}\n'
                '- product: e.g. {product}\n'
                '- creative-work: e.g. {creative_work}\n'
                '- group: e.g. {group}\n'
            )
            self.picl_prompt2 = (
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.fusion_prompt = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- person: e.g. {person}\n'
                '- location: e.g. {location}\n'
                '- corporation: e.g. {corporation}\n'
                '- product: e.g. {product}\n'
                '- creative-work: e.g. {creative_work}\n'
                '- group: e.g. {group}\n'
                'Here are some examples:\n'
            )
        elif self.dataset == 'ACE2004' or self.dataset == 'ACE2005':
            self.baseline_prompt = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- person\n'
                '- location\n'
                '- organization\n'
                '- geographical-social-political\n'
                '- weapon\n'
                '- facility\n'
                '- vehicle\n'
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.icl_prompt1 = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- person\n'
                '- location\n'
                '- organization\n'
                '- geographical-social-political\n'
                '- weapon\n'
                '- facility\n'
                '- vehicle\n'
                'Here are some examples:\n'
            )
            self.icl_prompt2 = (
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.picl_prompt1 = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- person: e.g. {person}\n'
                '- location: e.g. {location}\n'
                '- organization: e.g. {organization}\n'
                '- geographical-social-political: e.g. {geographical_social_political}\n'
                '- weapon: e.g. {weapon}\n'
                '- facility: e.g. {facility}\n'
                '- vehicle: e.g. {vehicle}\n'
            )
            self.picl_prompt2 = (
                'You should output your results in the format {"type": [entity]} as a json.'
            )

            self.fusion_prompt = (
                'Please list all named entities of the following entity types in the input sentence:\n'
                '- person: e.g. {person}\n'
                '- location: e.g. {location}\n'
                '- organization: e.g. {organization}\n'
                '- geographical-social-political: e.g. {geographical_social_political}\n'
                '- weapon: e.g. {weapon}\n'
                '- facility: e.g. {facility}\n'
                '- vehicle: e.g. {vehicle}\n'
                'Here are some examples:\n'
            )

class PointICL:
    def __init__(self, dataset, type2entity, point_entity_cnt, use_bert=False):
        self.dataset = dataset
        self.type2entity = type2entity
        self.cnt = point_entity_cnt
        self.use_bert = use_bert
        self.point_entity = {}
    
    def get_point_entity(self):
        if self.dataset == 'CoNLL2003':
            if self.use_bert:
                if self.cnt == 5:
                    self.point_entity = {} # K-Means results
                elif self.cnt == 10:
                    self.point_entity = {} # K-Means results
            else:
                self.point_entity = {
                    'per': random.sample(self.type2entity['PER'], self.cnt),
                    'org': random.sample(self.type2entity['ORG'], self.cnt),
                    'loc': random.sample(self.type2entity['LOC'], self.cnt),
                    'misc': random.sample(self.type2entity['MISC'], self.cnt)
                }
        elif self.dataset == 'WNUT2017':
            if self.use_bert:
                if self.cnt == 5:
                    self.point_entity = {} # K-Means results
                elif self.cnt == 10:
                    self.point_entity = {} # K-Means results
            else:
                self.point_entity = {
                    'person': random.sample(self.type2entity['person'], self.cnt),
                    'location': random.sample(self.type2entity['location'], self.cnt),
                    'corporation': random.sample(self.type2entity['corporation'], self.cnt),
                    'product': random.sample(self.type2entity['product'], self.cnt),
                    'creative-work': random.sample(self.type2entity['creative-work'], self.cnt),
                    'group': random.sample(self.type2entity['group'], self.cnt)
                }
        elif self.dataset == 'ACE2004' or self.dataset == 'ACE2005':
            if self.use_bert:
                if self.cnt == 5:
                    if self.dataset == 'ACE2004':
                        self.point_entity = {} # K-Means results
                    elif self.dataset == 'ACE2005':
                        self.point_entity = {} # K-Means results
                elif self.cnt == 10:
                    if self.dataset == 'ACE2004':
                        self.point_entity = {} # K-Means results
                    elif self.dataset == 'ACE2005':
                        self.point_entity = {} # K-Means results
            else:
                self.point_entity = {
                    'person': random.sample(self.type2entity['person'], self.cnt),
                    'location': random.sample(self.type2entity['location'], self.cnt),
                    'organization': random.sample(self.type2entity['organization'], self.cnt),
                    'geographical-social-political': random.sample(self.type2entity['geographical-social-political'], self.cnt),
                    'weapon': random.sample(self.type2entity['weapon'], self.cnt),
                    'facility': random.sample(self.type2entity['facility'], self.cnt),
                    'vehicle': random.sample(self.type2entity['vehicle'], self.cnt)
                }
        
        return self.point_entity    
