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
                    self.point_entity = {
                        'per': ['Peter Munk', 'Pablo Campana', 'Mark Thompson', 'David Nainkin', 'Stanley'],
                        'org': ['Deportivo Espanol', 'T&N Plc', 'Burnley', 'Reuters Television', 'Montreal'],
                        'loc': ['Jumet', 'Torshavn', 'Venezuela', 'Nottingham', 'Montreal'],
                        'misc': ['Ennea', 'Venezuelan', 'EURO CUP', 'Bedouin Palestinians', 'England-based']
                    }
                elif self.cnt == 10:
                    self.point_entity = {
                        'per': ['Peter Munk', 'Mark Thompson', 'Benoit Cauet', 'Pavel Buran', 'Mohamed Fahim Rayyan', 'Carlos Secretario', 'Stanley', 'Franco Baresi', 'David Gilford', 'Mark Dekker'],
                        'org': ['Chiron', 'Cincinnati Reds', 'Baxter International Inc', 'Exeter', 'PEC', 'ZIFA', 'Nottingham Forest', 'FK Jablonec', 'Council of Europe', 'Greenville'],
                        'loc': ['Taiwan Strait', 'San Francisco', 'Paris', 'Karak', 'Jeddah', 'Michigan', 'MANCHESTER', 'Nottingham', 'Spitzbergen', 'Venezuela'],
                        'misc': ['American League Western', 'Croatian', 'Hitleresque', 'Euro 96', 'GERMAN OPEN', 'Palestinian-ruled', 'Palestinian Arabs', 'Moroccan', 'Adriatic', 'European Champions Cup']
                    }
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
                    self.point_entity = {
                        'person': ['Michael Vick', 'Kanye West', 'Ben Bishop', 'Mark Rowan', 'Roger Waters'],
                        'location': ['Downtown La Grange', 'the Apollo', 'The Scoot Inn', 'Potbelly Lincoln Park', 'San Francisco'],
                        'corporation': ['Jason Industries', 'Citizens Bank', 'Evergreen Subaru', 'sun news', 'Nex-Tech Wireless'],
                        'product': ['Halo Reach', 'PlayStation Network', 'Panasonic AG-HMC40P Professional 3mos AVCCAM', 'Vanilla vodka', 'Jaket Korea BARCA " RIDE "'],
                        'creative-work': ['We All Are Dancing', 'The Devils Art', 'Last Christmas', 'Princess Lover OVA 1', 'rocky horror show'],
                        'group': ['Ryerson Quidditch', 'The New World Order', 'Maroon 5', 'Gym Class Heroes', 'Chicago Blackhawks']
                    }
                elif self.cnt == 10:
                    self.point_entity = {
                        'person': ['Mark Rowan', 'Mannie Fresh', 'Justin Bieber', 'Catherine Olson', 'Steve King', 'Adam Beyer', 'Eduardo Surita', 'Michael Owen', 'Ben Bishop', 'Jesus Christ'],
                        'location': ['Hyde Park Corner', 'Glover Park', 'San Jose', 'U.S. Capitol building', 'VISIONS LOUNGE', 'San Francisco', 'CAFE NINE', 'The Causeway', 'Arthurs Pass', 'Ramen Kan Bondi Junction'],
                        'corporation': ['Prana Recovery', 'Huffington Post', 'Serious Eats', 'Nex-Tech Wireless', 'Jason Industries', 'High House Farm', 'Pizza Hut', 'Super City', 'Game Informer', 'Evergreen Subaru'],
                        'product': ['Vanilla vodka', 'server 2008 R2', 'Zune HD', 'BOGEN / MANFROTTO 501 HEAD with camera mounting plate', 'Enslaved : Odyssey to the West', 'Flip MinoHD', 'Playstation 4', 'Sandman Slim', 'Club Penguin', 'Lincoln park after dark'],
                        'creative-work': ['Breaking Dawn', 'The Inbetweeners', 'We All Are Dancing', 'The Weekend Grind', 'Avengers 3', 'Last Christmas', 'Figgy Soda', 'Atlas Shrugs', 'S8 blog', 'A Song Of Ice &amp; Fire'],
                        'group': ['Ryerson Quidditch team', 'Green Day', 'Crown College', 'Wright family', 'Baka Boyz', 'Younger Brother', 'The Wolves', 'The Muppets', 'WB WILDCATS', 'Philadelphia Eagles']
                    }
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
                        self.point_entity = {
                            'person': ['holly firfer', 'Sen . John McCain', 'President Castro', 'spokeswoman for the Seattle field office of the FBI', 'travelers'],
                            'location': ['southwest China', 'inland', 'the border between Lebanon and Israel', 'the ocean', 'the west bank'],
                            'organization': ['amtrak', "the corporations with the most profits among Tianjin ' s foreign merchant investment corporations", 'the naacp', 'the utility', 'the Reagan White House'],
                            'geographical-social-political': ['the Chinese and Russian governments', 'the state', 'venezuela', 'Moroccan', 'Portland , Oregon'],
                            'weapon': ['stones', 'israeli guns', 'the bomblets found', 'a 1 . 3 kg TNT time bomb', 'live ammunition'],
                            'facility': ['the court building', 'the Seattle zoo', 'buildings', 'the studio', 'a park in the shadow of an oil refinery'],
                            'vehicle': ['helicopters', 'the small boat carrying the explosives', 'the destroyer', 'a car', 'taxi']
                        }
                    elif self.dataset == 'ACE2005':
                        self.point_entity = {
                            'person': ['Gregory Lynch', 'travelers', 'the pro - U . N . lobby', 'the military officials', 'U . S . military officials'],
                            'location': ['the southern outskirts of the Iraqi capital', 'us', 'the Baghdad area', 'the midwest', 'the Plains'],
                            'organization': ['the hospital chains', 'cnn', "the world ' s largest airline", 'The FBI', 'forces'],
                            'geographical-social-political': ['the state of Iraq', 'syria', 'iraqi', 'Montreal', 'orlando , florida'],
                            'weapon': ['medium - and long - range missiles', 'the rifle', 'biological weapons', 'mortar', 'weapons'],
                            'facility': ['the ports', 'saddam hussein international airport , just on the outskirts of baghdad', 'the command center', 'the baghdad airport', 'which'],
                            'vehicle': ['plane', 'the Zodiac boat', 'a u . s . tank', 'U . S . combat helicopters', 'a tank']
                        }
                elif self.cnt == 10:
                    if self.dataset == 'ACE2004':
                        self.point_entity = {
                            'person': ['The Syrian leader', 'the Israeli foreign minister', 'Melvin', 'Vice - President Al Gore', 'journalists', 'hillary', 'The lawyer', 'the 58 - year - old man who becomes president of mexico tomorrow , strikingly handsome , wily in the ways of modern politics , enormously appealing as a symbol of change', 'Sen . John McCain', 'Eddie Mair'],
                            'location': ['every region of the world', 'the west bank', 'The area', 'inland', 'the ocean', 'the southern part of the country', 'the cave', 'central florida', 'the Jordanian border', 'the mediterranean coast'],
                            'organization': ['ABC News', 'the university', 'Rutgers', 'separatist organization Eta', 'the Navy', 'Multinational companies investing in China', "the corporations with the most profits among Tianjin ' s foreign merchant investment corporations", 'the Florida high court', 'BBC', 'the Reagan White House'],
                            'geographical-social-political': ['the state', 'Portland , Oregon', 'venezuela', 'Beihai', 'Moroccan', 'the west bank town of hebron', 'the US government', 'Michigan', 'the Chinese capital', 'Paris'],
                            'weapon': ['helicopter missiles', 'a slingshot', 'it', 'israeli guns', 'gunpowder', '368', 'the bomblets found', 'the same chemical used in a Tokyo subway attack in 1995 that killed 12 people', 'three missiles fired from an Apache helicopter', 'gun'],
                            'facility': ['office buildings', "Africa ' s largest hospital", 'the mansion', 'the plants', 'a park in the shadow of an oil refinery', 'offices', 'the control tower', 'the city center', 'the Nanchang freight port', 'which'],
                            'vehicle': ['ambulances', 'automobiles', 'the Russian submarine , Kursk , lying at the bottom of the Barents Sea', 'the destroyer', 'A Singapore Airlines 747 flying from Taiwan to Los Angeles', 'Russian submarine', 'the ship', 'a school bus in Gaza', 'a car', 'train']
                        }
                    elif self.dataset == 'ACE2005':
                        self.point_entity = {
                            'person': ['scott', 'kelly wallace', 'astronauts', 'the pro - U . N . lobby', 'the three assets', 'former president bill clinton', 'the former iraqi leader', 'Thieves', 'U . S . and British forces', 'the pope'],
                            'location': ['the baghdad district', 'central florida', 'mesopotamia', 'the southern outskirts of the Iraqi capital', 'the Iraqi border', 'the iraqi border', 'western Iraq', 'the Plains', 'anywhere in the world', 'the island'],
                            'organization': ['the justice department', "the world ' s largest airline", 'the Navy', 'the hospital chains', 'firm', 'cnn', "Saddam ' s secret police", 'The World Bank', 'the U . S . Army', 'the conglomerate'],
                            'geographical-social-political': ['the Kremlin', 'Michigan', 'orlando , florida', 'the town of Karbala', 'Our', 'serbian', 'the city', 'the united states', 'syria', 'Montreal'],
                            'weapon': ['grenade launchers', 'the laser', 'an American bomb', 'nuclear weapons', 'medium - and long - range missiles', 'a smoking gun', 'Grenades', 'bullet', 'nuclear arms', 'its weapons'],
                            'facility': ['the Kazimiya mosque in the northeast of Baghdad', 'the Washington Monument', 'the courthouse in New York', 'these', 'an iraqi base in an unused factory in the southern suburbs of that city', 'the hospital', 'Saddam International Airport', 'the baghdad airport', 'the street', 'runways'],
                            'vehicle': ['an f - 16', 'which', 'a u . s . tank', 'the airplane itself', 'airline', 'the tank', 'helicopters', 'the mars polar lander', 'the military vehicles', 'this chopper']
                        }
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
