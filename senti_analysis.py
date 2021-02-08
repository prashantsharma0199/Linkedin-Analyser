from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import matplotlib.pyplot as plt
from textblob import TextBlob 
import os
import time
import string
import pandas as pd
from nltk.corpus import stopwords



my_name = input("Enter your name: ")
f_name = input("Enter the name of your data file you need to analyse (without file extension): ")

print("Please wait your report is generating.....")

analyzer = SentimentIntensityAnalyzer()

folder="report/"+my_name
if not os.path.exists(folder):
    os.makedirs(folder)


time.sleep(5)
data=open("report/"+my_name+"/Report-"+my_name+".txt","w+",encoding="utf-8")


#Calculating number of lines of file
fname="data/"+f_name+".txt"
num_lines = 0
with open(fname, 'r',encoding="utf-8") as f:
    for line in f:
        num_lines += 1
# print("Number of lines:")
# print(num_lines)


#Opening and reading lines of the file
f=open(fname, 'r',encoding="utf-8")
line=f.readlines()

polarity=[]
subjectivity=[]
pos=0
neg=0
neu=0

#Sentimental analysing using Vader and writing the report
for i in range(0,num_lines,2):
	text=line[i]
	# print(text+"\n")
	vs=analyzer.polarity_scores(text)
	
	blob=TextBlob(text)
	pol= blob.sentiment.polarity
	sub= blob.sentiment.subjectivity

	polarity.append(pol)
	subjectivity.append(sub)

	print("sentence was rated as ", vs['neg']*100, "% Negative")
	print("sentence was rated as ", vs['neu']*100, "% Neutral") 
	print("sentence was rated as ", vs['pos']*100, "% Positive")

	if vs['compound'] >= 0.05 : 
		print("Positive")
		pos=pos+1 
  
	elif vs['compound'] <= - 0.05 : 
		print("Negative")
		neg=neg+1
  
	else : 
		print("Neutral")
		neu=neu+1

	# print(vs)
	data.write(text+"Overall Sentiment Analysis: ")
	data.write(json.dumps(vs))
	data.write("\n")

pos_per=(pos/num_lines) * 100
pos_per= round(pos_per,2)

neg_per=(neg/num_lines) * 100
neg_per= round(neg_per,2)

neu_per=(neu/num_lines) * 100
neu_per= round(neu_per,2)



data.write("From ")
data.write(json.dumps(num_lines))
data.write(" captions:\n")

data.write("percentage of captions that are ")
a_dictionary = {"positive(%)" : pos_per, "negative(%)" : neg_per, "captions are neutral(%)": neu_per}
str_dictionary = repr(a_dictionary)

data.write("Results: " + str_dictionary + "\n")

#report generated
data.close()

# print(polarity)
# print("\n")
# print(subjectivity)

scat_name="report/"+my_name+"/Scatter-"+my_name+".png"
pie_name="report/"+my_name+"/Pie-"+my_name+".png"

plt.grid()
plt.scatter(polarity, subjectivity)
plt.title("Sentimental Analysis")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.savefig(scat_name)
plt.show()


labels = ['Neutral', 'Positive', 'Negative']
sizes = [neu_per,pos_per,neg_per] 
explode = (0, 0, 0)  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax1.axis('equal')  
plt.tight_layout()
plt.savefig(pie_name)
plt.show()


print("Your report has been generated !!")


#Complete ocean_analysis


file = open("report/"+my_name+"/Report-"+my_name+".txt",'r',encoding='utf-8')


text = file.read()
file.close()



def calcy(tokens, listy):
	counter=0
	for word in tokens:
		for text in listy:
			if (word == text):
				counter=counter+1;
	
	return counter;


# splitting into tokens
tokens = text.split()

# removing punctuation
table = str.maketrans('', '', string.punctuation)
tokens = [w.translate(table) for w in tokens]
    
# removing tokens that are not alphabetic
tokens = [word for word in tokens if word.isalpha()]
    
# filtering out stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]

# filtering out short tokens
tokens = [word for word in tokens if len(word) > 1]

#converting to lowercase
tokens = [word.lower() for word in tokens]

# print(tokens)
lengy=len(tokens)



#Neuroticism
neuro = ['awful','though','lazy','worse','depressing','irony', 'road', 'terrible', 'southern', 'stressful', 'horrible',
	'sort', 'visited', 'annoying', 'ashamed', 'ground', 'ban', 'oldest', 'invited', 'completed', 'firmness', 'strength',
	'soundness', 'durability', 'permanence', 'solidity', 'constancy', 'steadiness', 'steadfastness', 'frailty', 'fragility',
	'unpredictability', 'unreliability', 'fickleness', 'unsteadiness', 'inconstancy', 'changeableness', 'calmness', 
	'peace','mind','balance' , 'common', 'sense', 'mental', 'health', 'ability','remain', 'balanced', 'calm', 'stable', 
	'absence', 'illness', 'affective','tolerance', 'balanced', 'being' ,'free', 'agitation', 'disturbance', 'cognitive',
	'power', 'emotional', 'health', 'resilience', 'robustness', 'strength', 'strengths', 'freedom', 'disturbance', 
	'health', 'healthy', 'psychological', 'lucidity', 'lucidness', 'balance', 'equilibrium', 'faculties', 'soundness',
	'stability', 'patience', 'stability', 'well', 'being', 'rationality', 'reason', 'reliable' ,'steadiness', 'temperament', 
	'right', 'mindedness', 'saneness', 'sanity', 'serenity', 'sound', 'soundness', 'emotions', 'stability', 'mind', 'mindset',
	'steadiness','temperament', 'diseases', 'illness', 'disorder', 'balance', 
	'behavioral', 'calm', 'clear', 'headedness', 'cognitive' , 'faculties', 'composure', 'dispassion', 'doldrums', 
	'judgment', 'sense', 'human', 'heterotypic' ,'stability', 'hush', 'impassivity', 'imperturbation', 'inner' ,'balance', 
	'intellectual' ,'capacity', 'mellowness', 'fitness', 'good', 'harmony', 'healthcare', 'wellness', 'normal',
	'normality', 'peace', 'peacefulness', 'persistence', 'placidity', 'presence', 'psychiatric', 'psychic', 'balance', 
	'coherence', 'psychosocial', 'well-being', 'quiet', 'quietness', 'rationality', 'rationality', 'reasonableness', 'repose',
	'rest', 'restraint', 'robustness', 'sane mind', 'sense', 'silence', 'sound','though', 'sunset', 'drop', 'combined', 'feeling',
	'judgment', 'stability', 'stillness', 'stoicism', 'tranquility', 'awful', 'sick', 'road', 'ground', 'terribly', 'cranky', 'stress', 
	'feeling', 'southern', 'stressful', 'myself', 'though', 'feel', 'sweater', 'county', 'scenario', 'ashamed', 'feels', 
	'oldest', 'spoiled', 'sick', 'later', 'yay', 'road', 'possibly', 'completely', 'thirty', 'though', 'poem', 'wild', 
	'desperately', 'pregnancy', "shouldn't", 'lazy', 'refuse', 'irony', 'pretend', 'visited', 'horrible', 'harsh', 'combined', 
	'stupid', 'uncomfortable', 'though', 'fuck', 'drugs', 'guardian', 'sizes', 'smoke', 'city', 'Irish', 'messy', 'football', 'wife', 
	'silly', 'street', 'easier', 'opinions', 'lazy', 'shorter', 'expecting', 'mountain', 'fit', 'al', 'instead', 'realistic', 
	'fire', 'apart', 'drops', 'already', 'lazy', 'awful', 'bull', 'Southern', 'uncomfortable', 'lately', 'myself'] 

#Agreeableness
agree= ['wonderful', ' together', 'visiting', 'morning', 'spring', ' porn', 'walked', 'beautiful', 'staying', 'felt', 'cost', 'share', 
	'gray', 'joy', 'afternoon', 'day', 'moments', 'hug', 'glad', 'fuck', 'space', ' anger', 'numbers', ' home', 'leisure', 'time', 'motion',
	' up', 'family', ' death', 'positive', 'emotions', 'down', 'negative', 'emotions', 'optimism', 'inclusive', 'swearing', 'sports',
	'causation', 'time', ' home', ' swearing', ' anger', 'motion', ' leisure', ' family', 'up', 'down', 'numbers', ' positive', 'emotions',
	'inclusive', 'grooming', 'negative', 'emotions', ' space', 'optimism', 'death', 'anger', 'optimism', 'leisure', ' swearing', ' positive',
	'emotions', ' motion', 'space', 'family', 'inclusive', 'home', 'up', 'down', 'tentative', 'death', ' sports', 'causation', 'time',
	'idiot', ' hug', 'blast', ' chips', ' greeted', ' minutes', 'rest', ' times', ' cup', 'beach', ' solved', 'seconds', 'olympic',
	'stupid', 'following', 'dinner', 'participants', 'die', 'fabulous', 'sharing', 'cooperation', 'anger', ' swearing', 'space', 'numbers',
	'negative', 'emotions', ' optimism', ' inclusive', 'money', 'communication', 'death', 'hearing', 'prepositions', 'emotions', 
	'causation', 'negation', 'motion', 'home', 'time','wonderful', ' together', 'visiting', 'morning', 'spring', ' porn', 'walked',
	'beautiful', 'staying', 'felt', 'cost', 'share', 'gray', 'joy', 'afternoon', 'day', 'moments', 'hug', 'glad', 'fuck', 'space', 
	' anger', 'numbers', ' home', 'leisure', 'time', 'motion', ' up', 'family', ' death', 'positive', 'emotions', 'down', 'negative', 
	'emotions', 'optimism', 'inclusive', 'swearing', 'sports', 'causation', 'time', ' home', ' swearing', ' anger', 'motion', ' leisure',
	' family', 'up', 'down', 'numbers', ' positive', 'emotions', 'inclusive', 'grooming', 'negative', 'emotions', ' space', 'optimism', 'death',
	'anger', 'optimism', 'leisure', ' swearing', ' positive', 'emotions', ' motion', 'space', 'family', 'inclusive', 'home', 'up', 'down', 
	'tentative', 'death', ' sports', 'causation', 'time', 'idiot', ' hug', 'blast', ' chips', ' greeted', ' minutes', 'rest', ' times', ' cup',
	'beach', ' solved', 'seconds', 'olympic', 'stupid', 'following', 'dinner', 'participants', 'die', 'fabulous', 'sharing', 'cooperation', 
	'anger', ' swearing', 'space', 'numbers', 'negative', 'emotions', ' optimism', ' inclusive', 'money', 'communication', 'death', 'hearing',
	'prepositions', 'emotions', 'causation', 'negation', 'motion', 'home', 'time', 'mad', 'anger', 'livid', 'irate', 'glare', 'snarl', 'quarrel',
	'temper', 'furious', 'incense', 'glower', 'pissed', 'heated', 'sore', 'scowl', 'rile', 'cross', 'infuriate', 'upset', 'huff', 'madden', 'storm', 
	'enrage', 'choleric', 'steam', 'fume', 'flame', 'wrangle', 'retort', 'tirade', 'moody', 'ropable', 'gram', 'steaming', 'altercation', 'provoke',
	'wrathy', 'offend', 'annoy', 'growl', 'hot', 'hopping','mad', 'see red', 'wroth', 'fit to be tied', 'pique', 'apoplectic', 
	'rampage', 'rise', 'snaky', 'fit', 'irascible', 'wrathful', 'ballistic', 'stuffy', 'argument', 'seethe', 'strife', 'bad-tempered', 'stomach', 
	'annoyed', 'quick-tempered', 'fly off the handle', 'steamed', 'mob', 'het up', 'red', 'scold', 'black', 'go through the roof', 'rage', 'incandescent',
	'outrage', 'flare', 'bicker', 'furiously', 'huffy', 'boil', 'teed off', "blow one's top", 'on the warpath', 'fight', 'maggoty', 'cheesed off', 'hiss',
	'irritable', 'rant', 'placate', 'irritate', 'waxy', 'snuffy', 'even-tempered', 'irritated', 'lowering', 'flounce', 'piss off', 'bitter', 'row', 'exasperate', 
	'incensed', 'yond', 'word', 'touchy', 'shirty', 'wrath', 'static', 'flip', 'ratty', 'rave', 'frost', 'inflame', 'sharp', 'blow-up', 'splenetic', 'earful', 
	'sizzle', 'go spare', 'bristle', 'foam at the mouth', 'dark', 'harangue', 'up in arms', 'angrily', 'scream', 'acrimonious', 'mollify', 'berate', 'murderous',
	'indignant', 'red-faced', 'main', 'go ape', 'snap', 'worked up', 'brown', 'sorehead', 'stalk', 'soothe', 'bark', 'tetchy', 'have a cow', 'bait', 'irritation', 
	'argue', 'foam', 'lecture', 'fiery', 'white-hot', 'hotly', 'blackly', 'hot-tempered', 'crazy', 'spite', 'hullabaloo', 'wild-eyed', 'stroppy', 'get',
	'aggrieved', 'sulky', 'jealous', 'burn', 'upsetting', 'ornery', 'conflict', 'snit', 'aggro', 'uptight', 'lose your temper', 'do your nut', 'mood',
	'awm', 'aym', "hornet's nest", 'gall', 'flare-up', 'savage', 'blow', "make one's blood boil", 'temperamental', 'runâ€“in', 'grill', 'crook', 'hangry',
	'frustrated', 'ruckus', 'lower', 'blow', 'fuse', 'gasket', 'lose your rag', 'cool it', 'short-tempered', 'dudgeon', 'combust', 'volatile', 'flaming',
	'maddening', 'iracund', 'rankle', 'hopping', 'tee someone off', 'hit the roof', 'go up the wall', 'blast', 'hassle', 'forgive', 'wrangler', 'sputter',
	'in a huff', 'tick', 'bile', 'see', 'nettle', 'embitter', 'rating', 'fly into a rage', 'bent out of shape', 'pitiful', 'stabby', 'blowup', 'blow a fuse', 
	'dyspeptic', 'sullen', 'prickly', "make someone's hackles rise", 'fury', 'roasting', 'enraged', 'steam up', 'cool', 'boiling', 'be jumping up and down',
	'go off the deep end', 'inveigh', 'incite', 'hothead', 'unpleasant', 'fuck', 'abuse', 'spit', 'blood', 'venom', 'quickâ€“tempered', 'virago', 'belligerent',
	'flip out', 'clash', 'froth at the mouth', 'thunderous', 'fire', 'ugly', 'cut up', 'patience', 'pacify', 'rhubarb', 'blow a gasket', 'overheat', 
	'annoyance', 'tantrum', 'go bananas', 'glaring', 'blood', 'sparks fly', 'calm', 'resentful', "keep one's shirt on", 'gut', 'bitterly', 'quick temper',
	'swear', 'broil', 'darken', 'lose it', 'cut up rough', 'uproar', 'earbashing', 'gorge', 'conciliate', 'go off on one', 'tee', 'spare',
	 'a red rag to a bull', 'infuriated', 'hatter', 'blazing', 'galsome', 'temperament', 'stir', 'conniption', 'offense', 'incandescent with rage',
	  'frown', 'brooding', 'bridle', 'inflammatory', 'dander', 'throw', 'fit', 'sicken', 'simmer down', 'nut', 'raise hell', 'for the love of god',
	   'hissy fit', 'furor', 'crackers', 'ropeable', 'riley', 'affront', 'go postal', 'wring someoneâ€™s neck', 'fight like cat and dog', 'vituperative',
	    'ill-tempered', 'tooshie', 'ticked', 'back', 'breathing fire', 'bealing', 'stunt', 'howl', 'madder', 'madding', 'high words', 'foul', 'ablaze', 
	    '(as) mad as a meat axe', 'slanging match', 'brawl', 'contentious', 'embittered', 'run high', 'go to hell', 'roar', 'hurl', 'tartar', 'thunder',
	     'rejoinder', 'curse', 'displease', 'spitfire', 'give someone a mouthful', 'soft', 'imprecation', 'crank', 'tempers flare', 'trouble', 'disgruntled', 
	     'tee sb off', 'pissed off', 'sulk', 'vituperation', 'vitriol', 'be foaming at the mouth', 'bate', 'blistering', 'disorderly', 'throat', 
	      'rate', 'burn someone up', 'lose', 'cool', 'composure', 'go nuts', 'propitiate', 'cantankerous', 'wreakful', 'filthy', 'honked off']
       

#Openness
openn= ['novel', 'fame', 'urge', 'decades', 'urban', '8th', 'glance', 'length', 'poetry', 'literature', 'audience', '8', 'anniversary',
	'6', 'loves', 'narrative', 'lines', 'bears', 'thank', 'humans', 'beauty', 'moon', 'blues', 'sky', 'plants', 'dance', 'beautiful', 
	'trees', 'planted', 'flowers', 'sang', 'blue', 'sings', 'danced', 'music', 'afterwards', 'tree', 'painted', 'hills', 'outdoor', 
	'feel', 'breathe', 'feeling', 'awful', 'stressful', 'stress', 'fabulous', 'felt', 'heart', 'lucky', 'cried', 'overwhelming', 
	'sleep', 'hours', 'scared', 'sick', 'therapy', 'am', 'myself', 'feels', 'streets', 'city', 'century', 'sexual', 'industry', 
	'businesses ', 'south ', 'tour ', 'sean ', 'global ', 'diaper ', 'immigration ', 'countries ', 'legal ', 'poet ', 'buildings ', 
	'employment', 'west ', 'little ', 'al ', 'against ', 'argument ', 'knowledge ', 'by ', 'sense ', 'political ', 'models ',
	'belief ', 'human ', 'historical ', 'greater ', 'state ', 'universe ', 'philosophy ', 'humans ', 'beings ', 'evidence ',
	'scientists ', 'thank ', 'leap', 'complicated ', 'literature ', 'particularly ', 'prayers ', 'giveaway ', 'thankful ', 
	'hubby ', 'let ', 'unlikely ', 'less ', 'complex ', 'folk ', 'terms ', 'fucking ', 'entirely ', 'structure ', 'cultural ', 
	'liberal ', 'university ', 'bizarre ', 'imaginative', 'fancy', 'fantasy', 'imaginary', 'visionary', 'fanciful', 'imagine',
	'fantastic', 'creative', 'earthbound', 'fiction', 'relive', "mind's eye", 'ideal', 'prosaic', 'fire', 'conceit', 'imagery',
	'phantasm', 'invention', 'vision', 'artistic', 'imaginational', 'pedestrian', 'misimagination', 'art', 'imaginal', 
	'dreamland', 'chimera', 'enterprising', 'creation', 'conceited', 'purblind', 'nonentity', 'phantasy', 'sterile', 'fertile',
	'fictive', 'dream up', 'hack', 'fictitious', 'create', 'poetry', 'makebelieve', 'brain', 'otherworldly', 'step', 'spin', 
	'invent', 'romantic', 'associate', 'artist', 'fine art', 'stodgy', 'dryness', 'unimaginative', 'dream world', 'project',
	'dream', 'association', 'fancier', 'fancied', 'exalt', 'imaginability', 'ingenuity', "in one's mind's eye", 'figment',
	'suspicion', 'romanticism', 'imaginatively', 'picture', 'imaginings', 'poet', 'arid', 'ideally', 'fume', 'devise', 'myth',
	'madeâ€“up', 'vivid', 'reproduce', 'image', 'mythical', 'conceive', 'inventive', 'foreshorten', 'soar', 'evoke', 'prosy',
	'vapid', 'cloudland', 'daydream', 'fertility', 'seize', 'verve', 'flight', 'presentive', 'fanciless', 'surmise', 'scene',
	'concoct', 'thinking', 'reproductive imagination', 'catch', 'creativity', 'suffocate', 'stick-in-the-mud', 'referent',
	'dreamworld', 'visualize', 'poem', 'flighty', 'impress', 'mind', 'strong', 'short-sighted', 'wit', 'boilerplate',
	'capture', 'whimsical', 'connection', 'run riot', 'lively imagination', 'species', 'prose', 'apprehension', 'slave', 
	'flame', 'obvious', 'long-sighted', 'fictional', 'think', 'woolgathering', 'fashion', 'sobriety', 'real', 'manufacture', 'imaginatory',
	'short-sightedness', 'realistic', 'belles-lettres', 'blockish', "in your mind's eye", 'uninspired', "all in one's head", 'cleverness',
	'inventiveness', 'extravagance', 'vista', 'occur', 'conjure', 'ideaphoria', 'stolid', 'oneirodynia', 'reason', 'cloud','cuckoo','land', 
	'imaginous', 'generic', 'imaginarily', 'realism', 'grab', 'explicit', 'clever', 'conception', 'inanity', 'ingenious', 'chimerical',
	'stretch', 'reality', 'figmental', 'flat-footed', 'fecundity', 'choke', 'disimagine', 'imaginable', 'blear', 'fantasied', 'desiccated',
	'make', 'fevered', 'kingdom', 'concept', 'oppress', 'originality', 'viewy', 'salacious', 'viable', 'toy', 'diddle', 'feat', 'sterility',
	'made-up', 'play', 'physical', 'delirium', 'fiddle', 'mind','blowing', 'antiseptic', 'intellection', 'uninformed', 'imaginator',
	'run', 'uncreative', 'terre', 'cook up something', 'exhilarate', 'travel', 'barmecide', 'active', 'create','dream', 'up' ,'something',
	'role','take', 'run' ,'away', 'ghost' ,'story', 'man', 'literal', 'quicken', 'reverie', 'strain', 'presentation', 'ideoplastic', 
	'figure', 'evocation', 'reconjure', 'titillate', 'quickness', 'fairyland', 'fable', 'conceptual', 'style', 'fascination',
	'pamper', 'vapor', 'artistry', 'rein', 'overactive', 'novel', 'overwrought', 'televise', 'wild', 'theory', 'come', 'revealing', 'awful', 
	'autoeroticism', 'sally', 'cook', 'world', 'practical', 'ideoplasty', 'flight' ,'fancy', 'compass', 'dummy', "in one's wildest dreams",
	'sober', 'intellect', 'vivid','imagination', 'cookie', 'cutter', 'dismal', 'challenge', 'stir', 'free', 'actual', 'exuberant', 'composition', 
	'reevoke', 'conceitless', 'inconceivable', 'generically', 'made up', 'establish', 'oppression', 'idea', 'plod', 'vicarious', 'unlock', 'heart', 
	'poetic', 'lump', 'head', 'myopia', 'funky', 'physically', 'trilling', 'the real world', 'shadow', 'pragmatize', 'literal','minded', 
	'creative writing', 'life', 'plus', 'by numbers', 'therianthropic', 'associative', 'ablaze', 'material', 'chasten', 'journeyman', 
	'imagination', 'fantasy', 'excursive', 'body', 'abreact', 'real life', 'the arts', 'platitude', 'virtual', 'blue','sky', 'phantasmagoria',
	'fertilize', 'crudity', 'subordinacy', 'peake', 'projection', 'improvise', 'bugaboo', 'representation', 'idealism', 'common', 'develop', 
	'euhemerism', 'perception', 'tickle', 'pictorial', 'dreamer', 'imagineer', 'unfettered', 'fascinate', 'stimulus', 'mythicist', 'literature', 
	'nourish', 'transformation', 'go', 'lateral','thinking', 'routineer', 'energy', 'grandeur', 'polite', 'enthusiasm', 'mindscape', 'manning', 
	'precinct', 'work', 'crayon', 'manned', 'phenomenon', 'arouse', 'stupid', 'humor', 'dupe', 'ideogenetic', 'hold', 'perish', 'opinion', 'men', 
	'lunarian', 'fecund', 'baseless', 'clamber', 'schema', 'suborn', 'exert', 'centurion', 'infamous', 'feuerbach, ludwig', 'bubble', 'detector',
	'nozzle', 'illusion', 'loom', 'mount of luna', 'subtle', 'commodity', 'analysis', 'ellipsis', 'time-slip', 'statistics', 'resonance', 'lime', 
	'lode', 'vast', 'googol', 'suggestive', 'poverty', 'formulaic', 'horology', 'malignant', 'nonexistent', 'gnome', 'factitious', 'fantasia', 
	'prankster','maxim', 'simile', 'fairy', 'delusion']

	#Conscientiousness
consc= ['fired', 'Roberts ', 'rough ', 'Hawaii', 'desperate ', 'routine ', 'tbsp ', 'vegetables ', 'garlic ', 'temperature ', 'carrots', 
	' melted snack ', 'salad ', ' popcorn ', ' days', 'terror', 'jail ', 'warm', 'enjoying ', 'with ', 'extreme ', 'cheese ', 'rest ', 'hr ',
	'14 ', 'intelligent ', 'deck ', 'bang ', 'pity', 'lots ', 'stack ', ' 8 ', ' finished ', 'pathetic ', 'visit ', 'stupid ', 'idiot ',
	'religious ', 'vain ', 'decent', 'wallet ', 'deny', 'rarely ', 'bloody', 'protest', 'utter ', 'contrary', 'shame', 'majority', 'soldiers',
	'drunk', 'politically', 'democracy', 'entirely ', 'practical ', 'ready', 'HR', 'rarely', 'boring ', 'quality', 'overcome ', " mom's ",
	'until', 'clever', 'Mexican ', ' pace ', 'challenging ', 'addition', 'anxious', 'jokes', 'paid ', 'self', 'pride', 'ego', 'proud', 
	'amour-propre', 'egoism', 'selfâ€“worth', 'inferiority complex', 'amour propre', 'self-worth', 'self-respect', 'high-flown', 'indignity',
	'humiliation', 'puff', 'greatness', 'idiolatry', 'self-respecting', 'dignified', 'abasement', 'place', 'respect', 'disesteem', 'regard',
	'self-estimation', 'uppity', 'puncture', 'dysthymia', 'massage', 'deflate', 'egotism', 'autotheism', 'pompous', 'value', 'conceit', 'narcissism',
	'esteem', 'gay pride', 'estimation', 'nosism', 'pique', 'shake', 'selves', 'supportive therapy', 'peg', 'proudly', 'humble',
	'significant other', 'take', 'modesty', 'let-down', 'social promotion', 'bloated', 'prize', 'honor', 'negatively', 'nervous breakdown',
	'firmness', 'consideration', 'estimable', 'misesteem', 'upstart', 'self-regard', 'reputation', 'abase', 'consequential', 'sell', 'hold',
	'honour', 'credit', 'codependency', 'cripple', 'respectable', 'undervalue', 'cheap', 'bumptious', 'autogenous', 'autonomy', 'soi-disant',
	'autogenetic', 'destroy', 'major depressive disorder', 'self-government', 'adoration', 'onanism', 'autolatry', 'overrate', 'possessed',
	'count', 'autogeneal', 'disparage', 'brass', 'consider', 'creditable', 'auto-', 'let', 'self-contained', 'self-righteous', 'cocky',
	'existimate', 'self-rule', 'launderette', 'self-command', 'sufficient', 'self-deceit', 'win', 'opinion', 'acceptation',
	'honorific', 'herself', 'collected', 'self-induced', 'set store by', 'valuable', 'selfism', 'hoity-toity', 'debase', 'ascesis',
	'admire', 'compliment', 'disrepute', 'aplomb', 'meritorious', 'testimonial', 'well-beloved', 'dignity', 'confidence', 'denial',
	'self-abuse', 'autarchy', 'autodidact', 'premium', 'respected', 'put on a pedestal', 'estimate', 'autogenic', 'self-abnegation',
	'reputable', 'yourself', 'self-assured', 'address', 'self-love', 'think', 'sufficiency', 'bless', 'adore', 'autogamy', 'myself',
	'himself', 'self-mastery', 'existimation', 'cockalorum', 'reckoning', 'self-taught', 'egotist', 'truism', 'homage', 'egotistic',
	'demean', 'solid', 'assured', 'autokinesy', 'self-control', 'automotive', 'confident', 'worship', 'fellow', 'have', 'judge',
	'self-restraint', "on one's feet", 'acquit', 'autofluorescence', 'automorphic', 'big', 'flatulent', 'reverence', 'considered',
	'good', 'coolness', 'egocentric', 'self-defence', 'phlegm', 'oneself', 'self-praise', 'look', 'account', 'temperance', 'lowly',
	'smirk', 'unself-consciously', 'recommendation', 'upgrade', 'overestimate', 'automath', 'pedestal', 'relegate', 'prepossessing',
	'pluviograph', 'unself-conscious', 'brash', 'magnify', 'unself-consciousness', 'admirable', 'abstentious', 'best', 'sober-minded',
	'tender', 'possession', 'self-deception', 'simper', 'self-protection', 'reaper-and-binder', 'abnegation', 'l', 'self-assurance',
	'empirical ego', 'disrespect', 'gem', 'inwardly', 'selfâ€“pollinate', 'stuffy', 'Honourable', 'downgrade', 'dainty', 'steem',
	'discredit', 'esteeming', 'deign', 'esteemed', 'autonomous', 'autoinduction', 'nibs', 'uncreate', 'self-made', 'self-pitying',
	'decadent', 'lonesome', 'sen', 'prima facie', 'SASE', 'friend', '-teria', 'adjoin', 'willing', 'self-admiration', 'willy',
	'self-determining', 'commune', 'timid', 'assurance', 'portentous', 'masturbation', 'aut-', 'self-sterile', 'destruct',
	'impotent', 'aware', 'precious', 'repute', 'closed', 'self-conscious', 'temperate', 'decadence', 'autohypnosis',
	'automatic', 'groceteria', 'biggity', 'sovereignty', 'uncool', 'autarky', 'conscious', 'go', 'swaraj', 'tribute',
	'vaunt', 'diffident', 'self-harm', 'barograph', 'alter ego', 'austere', 'smug', 'make', 'self-destruction', 'self-imposed',
	'perk', 'self-satisfied', 'conciliate', 'importance', 'upstanding', 'self-fertile', 'person', 'self-confidence', 'dedication',
	'selfâ€“satisfied', 'poised', 'self-centered', 'deem', 'self-aggrandizement', 'absenter', 'commendation', 'apprize', 'autotheist', 
	'Peacock', 'dogs', 'self-rising flour', 'forbearance', 'esteemable', 'overesteem', 'thermograph', 'hubris', 'itself', 'honorable', 
	'preexistimation', 'singly', 'egomania', 'self-abasing', 'jack-in-office', 'deference', 'worth', 'continent', 'sycophant', 'independent',
	'disappreciate', 'evil', 'rate', 'special', 'fame', 'snide', 'price', 'continency', 'autopoiesis', 'arrest', 'autometry', 'degraded', 
	'regardful', 'depreciate', 'care', 'shame', 'cheapen', 'underappreciate', 'mal vu', 'self-important', 'suspect', 'famous', 'estimableness',
	'overween', 'dishonorable', 'better', 'enËˆdearing', 'reckon', 'lift', 'exalt', 'equipensate', 'selfâ€“betrayal', 'selfâ€“contented',
	'contentment', 'devoting', 'self-sustenance', 'loved', 'self-directed', 'picture', 'opinionative',
	'havlagah', 'self-annihilation', 'gratulatory', 'subsisting', 'deportment', 'self-gratification', 'seed', 
	'scrutiny', 'automobile', 'ruling', 'respectful', 'autokinetical', 'relation', 'naughting', 'autoregulation', 
	'impotent', 'gratulation', 'exam', 'endeared', 'consequence', 'confrontation', 'conception',
	'caused', 'assumption', 'transcendental self', 'hookweed', 'selve', 'self-repulsive', 'self-devotement', 'self-charity', 
	'trust', 'movement', 'fertilizing', 'forgetting', 'involvement', 'admiration', 'autoregulating',
	'butt', 'autoctony', 'self-understanding', 'limiting', 'content', 'self-limiting', 'blue lucy', 'opinioned', 
	'autophagous', 'searching', 'rule', 'involution', 'autoregulated', 'auto-destruct', 'destroying', 
	'assurance', 'self-repairing', 'self-sustaining', 'self-commune', 'revelative', 'deceptive', 'sufficing',
	'corrective', 'satisfaction', 'logical', 'reproducing', 'trivial', 'autofecundation', 'professed', 
	'carpenter weed', 'styled', 'pleased', 'involved', 'elected', 'collected', 'integral cover',
	'indulgent', 'bootstrap', 'autodidacticism', 'government', 'well-graced', 'stature', 'admiration', 'prestige', 'fallen',
	'behight', 'disgrace', 'eminence', 'dignation', 'greekling', 'build up', 'standing', 'self-dependent', 'impotency', 'self-assertive',
	'self-denying', 'impotence', 'asceticism', 'self-enjoyment', 'masturbatory', 'self-trust', 'self-stick', 'inconsistent', 'autolysis',
	'BSE', 'self-imposture', 'abstinence', 'self-abandonment', 'healâ€“all', "throw one's weight about (or around)", 'small self-administered pension scheme',
	'Se', 'congratulatory', 'feeling', 'flattering', 'motive', 'propelling', 'speaking rod', 'incontinence',
	'self-pity', 'self-ignorant', 'attitude', 'axiomatic', 'automatic door', 'amok', 'braggart', 'complacent', 'disinterested', 'distemper',
	'heal-all', 'jaunty', 'micromania', 'monastic', 'paradoxical', 'warm', 'self-willed', 'rectitudinous', 'bashful', 'collect', 'free-living',
	'increated', 'SIW', 'incontinent', 'heautophany', 'toplofty', 'obstinate', 'self-absorbed', 'self-depreciation', 'self-doubt', 'notself',
	'self-seed', 'shy', 'unity', 'selfâ€“governed', 'self-depending', 'self-reproaching', 'SRO', 'pushful', 'bounce', 'causeless', 'soppy',
	'sorry for oneself', 'shit-eating', 'cafeteria', 'non-', 'doubting', 'egotheism', 'murder', 'restraining', 'serve', 
	'seeker', 'ruth', 'being', 'autoload', 'aw-shucks', "be one's own boss", 'anemograph', 'self-devised', 'self-centering', 'lame',
	'inner space', 'locomotive', 'selfness', 'self-', 'grip', 'front', 'restrained', 'selfâ€“preservative', 'pushing', 'selfâ€“affected', 
	'cocker', 'selfâ€“knowing', 'betake', 'autokinetic', 'locknut', 'effete', 'pamper', 'imprudent', 'sentient', 'self-contempt',
	'self-criticism', 'self-fulfilling', 'self-justification', 'self-torture', 'selfie', 'unsure', 'self-supporting', 'apperception', 
	'selfâ€“identical', 'benign', 'soiâ€“disant', 'washateria', 'dumb', 'alone', 'assertive', 'allheal', 'auto-portrait', 'self-concern',
	'by-interest', 'philauty', 'masquerade', 'municipal', 'obey', 'primitive', 'vainglorious', 'selfâ€“repugnant', 'unction', 'denied',
	'hair shirt', 'selfâ€“confiding', 'headstrong', 'control', 'purdy', 'masterful', 'motor', 'oat', 'outrecuidance', 'paddle', 'conscience',
	'sober', 'tale', 'three-hanky', 'sang-froid', 'nainsel', 'hercogamous', 'SAE', 'selfâ€“revealing', 'unselfconscious', 'travel',
	'selfâ€“reflection', 'unbegotten', 'selfâ€“life', 'cock-a-hoop', 'boastful', 'bragging', 'ascetic', 'masturbate', 'locomotion', 
	'self-abhorrence', 'self-annihilated', 'self-applause', 'self-centring', 'increate', 'flatulence', 'self-delation', 'self-gratulation',
	'self-repugnant', 'axiom', 'SPM', 'carry', 'free', 'selfâ€“governance', "lose one's head", 'self-reflection', 'overblown', 'pontifical',
	'surety', 'self-consistent', 'oleaginousness', 'objective', 'unit', 'SLR', 'personally', 'diffidence', 'will', 'ownself', 'recover',
	'selfâ€“esteem', 'selfâ€“applauding', 'selfâ€“denying', 'debauched', 'easy', 'due', 'honorary', 'worthy', 'weigh', 'less', 'rank',
	'poise', 'selfâ€“actor', 'self-activity', 'expedient', 'furnish', 'interested', 'report', 'self-communion', 'ignoble', 'valued',
	'disconsider', 'admirably', 'expression', 'store', 'Jabir ibn Hayyan', 'honoris causa', 'selfâ€“sacrificer', 'booketeria',
	'self-devouring', 'jackeen', 'kamikaze', 'heartâ€“searching', 'pharisee', 'selfâ€“poised', 'Mushet steel', 'self-sealing',
	'snap', 'tin god', 'defenceless', 'lucrative', 'undisciplined', 'lave', 'gut', 'udomograph', 'fare', 'autotoxaemia', 'self-contradictory',
	'self-delusion', 'autodynamic', 'self-worship', 'self-interested', 'consistent', 'asocial', 'innermost', 'obscure', 'self-indulgent', 
	'self-regulating', 'selfish', 'ombrograph', 'fleshpots', 'face', 'dog', 'plain', 'self-approbation', 'self-serving', 'bouncy', 'perky', 
	'selfâ€“born', 'selfâ€“denier', 'selfâ€“incompatible', 'fanfaronade', 'complacency', 'truculent', 'dirigible', 'discipline', 'ill fame', 
	'postponement', 'mahatma', 'dear', 'blighter', 'sisterhood', 'revere', 'favorite', 'demonology', 'dime', 'ill repute', 'detraction', 'crow', 
	'navelâ€“gazing', 'bug', 'native', 'dramatize', 'discover', 'consciousness', 'knuckle', 'peevish', 'phlegmatic', 'self-consciously', 
	'self-flagellation', 'strut', 'seity', 'self-satisfying', 'bullish', 'barometrograph', 'ESS', 'self-explanatory', 'self-depreciatory', 
	'self-evaluation', 'self-parodic', 'heteropathic', 'sit', 'autocracy', 'autognosis', 'vile', 'accredit', 'aim', 'captivating', 'icon', 
	'superfly', 'marigraph', 'selfâ€“asserting', 'selfâ€“assured', 'selfâ€“important', 'crouse', 'chutzpah', 'self-consuming', 'bike', 'feisty',
	'composed', "drop one's bundle", 'crust', 'freedom', 'dogged', 'assure', 'atman', 'airship', 'insecure', 'embarrass', 'kip', 'luxurious', 
	'maintain', 'smugly', 'self-compatible', 'self-slaughter', 'holy', 'hubristic', 'hyetometrograph', 'intermeddle', 'self-inflicted', 
	'self-reprovingly', 'clasp lock', 'crack', 'cutesy', 'dogâ€“eatâ€“dog', 'donnism', 'autogamous', 'pushy', 'ruthful', 'pleasure', 'sel', 
	'selfâ€“appointed', 'disinvoltura', 'selfâ€“discrepant', 'selfâ€“realizationist', 'enforce', 'self-forgetful', 'self-incompatible', 
	'self-propagating', 'stay', 'soul', 'inmost', 'dandified', 'selfâ€“deceiver', 'self-sacrifice', 'prefer', 'disreputable', 'degrade', 
	'disaffect', 'status', 'salvo', 'excellently', 'arriviste', 'absquatulate', 'shrinking violet', 'self-seeking', "in one's own right", 
	'reseed', 'selfâ€“analytical', 'sordid', 'selfless', 'selfâ€“reflective', 'egoâ€“expansion', 'hedonism', 'individualism', 'Makarov', 
	'intrapersonal', 'disinterest', 'motorcar', 'assuming', 'selfâ€“disciplined','system', 'decorum', 'spit and polish', 'Method', 'methodical', 
	'social order', 'bundobust', 'decency', 'trimness', 'discipline', 'organization', 'anal', 'law of nature', 'logical', 'obedience', 'impious', 
	'Good Conduct Medal', 'piety', 'impiety', 'undutiful', 'obeisance', 'obey', 'attainment', 'capstone', 'triumph', 'strife', 'accomplishment', 
	'feat', 'stature', 'laureate', 'strive', 'aspirant', 'crown', 'deed', 'affectation', 'effort', 'agony', 'record', 'prize', 'kudos', 'AT', 
	'emulation', 'congratulate', 'exploit', 'unitary', 'laurel', 'masterpiece', 'landmark', 'emulate', 'nonachievement', 'performance', 'pinnacle', 
	'nisus', 'coup', 'work', 'credit', 'Grammy', 'award', 'success', 'reluctant', 'trophy', 'superhuman', 'preen', 'ï¿½clat [or] eclat', 'fight', 
	'mention', 'ambition', 'contention', 'norm', 'competition', 'Titan', 'behind', 'wrestle', 'agonistic', 'celebrated', 'avail', 'realization', 
	'accomplishment quotient', 'plateau', 'struggle', 'pursuit', 'atman', 'promise', 'slippage', 'distinction', 'achievement', 'compete', 'level', 
	'scholarship', 'achievance', 'Bronze Star', 'miracle', 'salute', 'glory', 'recognition', 'eminent', 'felicitate', 'acme', 'honorary', 
	'Pulitzer Prize', "feather in one's cap", 'ne plus ultra', 'Pyrrhic victory', 'monument', 'top', 'degree', 'candidacy', 'tour de force', 
	'thing', 'ambitious', 'Academy Award', 'rival', 'glorious', 'stroke', 'counterproductive', 'go', 'crowning', 'jihad', 'merit', 'supreme', 
	'systems analysis', 'standard', 'grade', 'summit', 'unambitious', 'scale', 'big-league', 'big league', "chef-d'oeuvre", 'oblique', 'letter', 
	'promising', 'medal', 'best', 'meritocracy', 'badge', 'hatchment', 'detract', 'aspiration', 'overtake', 'mediocrity', 'a feather in your cap', 
	'honor', 'tops', 'golden age', 'plan', 'frontier', 'precocious', 'weary', 'up', 'record-breaking', 'combative', 'strift', 'congratulation', 
	'panegyric', 'personal best', 'tension', 'genteel', 'reward', 'handicap', 'renowned', 'jest', 'surpass', 'vehicle', 'conative', 'feather', 
	'wishful', 'unlaborious', 'aspiring', 'dicty', 'straight', 'thorough', 'talent', 'aq', 'peak', 'educational quotient', 'result', 'prestige', 
	'repeat', 'fail', 'Eagle Scout', 'honor society', 'triumphant', 'magnum opus', 'no mean achievement/feat', 'chevisance', 'breakthrough', 'accredit', 
	'field', 'goal', 'abreast', 'rung', 'consummation', 'equivalency', 'excel', 'gain', 'kemp', 'tony', 'be gunning for', 'transcend', 'Goes', 'Clio', 
	'execution', 'be no mean feat', 'toil', 'futilitarian', 'liberation', 'hat-trick', 'walkover', 'gone', 'the acme', 'going', 'coat of arms', 
	'certification', 'full marks', 'intelligence test', 'trick', 'surname', 'spur', 'stellar', 'grade inflation', 'report', 'cap', 'asterisk', 
	'martial artistry', 'obstacle', 'impossibly', 'outshine', 'apex', 'batting average', 'match', 'bloom', 'on', 'sung', 'marshal', 'tenâ€“strike', 
	'feather in your cap', 'coping stone', 'plume', 'joint', 'limitation', 'trough', 'go one better', 'extraordinary', 'come', 'underachiever', 
	"a feather in one's cap", 'unaccomplished', 'poet laureate', 'exploiture', 'hurdle', 'long suit', 'shortcut', 'ante', 'MBE', 'ground', 
	'wish-fulfillment', 'walks of life', 'likely', 'emprise', 'Everest', "rest on one's laurels", 'epistemophilia', 'superiority complex', 
	'contestation', 'moderne', 'colluctancy', 'diligent', 'Church Militant', 'strain', 'citation', 'Triple Crown', 'job satisfaction', 
	'summa cum laude', 'Alumbrado', 'tug of war', 'make', 'sportsmanship', 'conflict', 'league table', "under one's belt", 'copestone', 'win', 
	'educational test', 'center of excellence', 'distinguished', 'belt', 'worst', 'illustrious', 'coming', 'famous', 'putty medal', 'ahead', 
	'phenomenon', 'vanity', 'flower', 'the norm', 'hack', 'limited', 'possibly', 'ARM', 'cloudward', 'geeâ€“whiz', 'frustrate', 'stunner', 
	'sadism', 'straight A', 'cock-a-hoop', 'acclaim', 'compensation', 'riband', 'luminary', 'giant', 'mediocre', 'heroine', 'resound', 'piss-poor', 
	'triple A', 'competitive', 'fulfilment', 'underachievement', 'have to hand it to someone', 'moral victory', 'famed', 'beamish', 'difficulty', 
	'foil', 'leading', 'swansong', 'signal', 'be level pegging', 'Platonic', 'conation', 'place', 'achievement quotient', 'crest', 'idealism', 
	'faustian', 'endeavour', 'conatus', 'Irish Republican Army', 'idealistic', 'aspirational voter', 'though/if I say it myself', 'renown', 'hurrah', 
	'get to/reach first base', 'masterstroke', 'halfway', 'accreditation', 'silver age', 'nadir', 'sank', 'strategically', 'welt', 'strike', 'colour', 
	'lock (something) up', 'Winnie', 'brilliant', 'success story', 'answerable', 'monopoly', 'registrar', 'bafta', 'give someone credit for', 
	'awards ceremony', 'film award', 'assessment', 'better', 'sink', 'niveau', 'ceremony', "that's not saying much", 'topflight', 'architect', 'point', 
	'banquet', 'hail sb/sth as sth', 'Emmy', 'build on sth', 'reliable', 'GPA', 'potentiality', 'beckon', 'slump', 'impressed', 'impossibilism', 
	'squirrel cage', 'Obie', 'clobbering machine', 'arrive', 'pride', 'Shingon', 'spare', 'battery', 'coat', 'objectivism', 'plane', 'renaissance', 
	'testimonial', 'honour', 'Edgar', 'Order of Canada', 'aid', 'fulfil', 'Air Medal', 'campaign', 'belted', 'felicitations', 'disenjoy', 'clinch', 
	'even', 'chair', 'Medal of Freedom', 'culture', 'inferior', 'improvement', 'milestone', 'elude', 'circle', 'victory', 'gratify', 'sunken', 
	'requisite', 'scorecard', 'triumphal', 'reputation', 'sinking', 'harvest', 'credit with', 'congratulatory', 'honors', 'hail', 'sunk', 
	'triumphantly', 'strong', 'of renown', 'Reuben', 'chronological age', 'second best', 'backâ€“patting', 'team', 'ability', 'crack', 'motto', 
	'valiant', 'earnestness', 'dance', 'quixotic', 'idealist', 'display', 'dark horse', 'career', 'Distinguished Flying Cross', 'tripleâ€“double', 
	'recognize', 'tincture', 'procurance', 'enterprise culture', 'front', 'certificate', 'the glass floor', 'term paper', 'lame duck', 'ton', 
	'qualification', 'group test', 'exertion', "Queen's Award", 'College Board', 'associate degree', 'closure', 'credential', 'strata', 'meed', 
	'hero', 'solifidian', 'subsizar', 'way to go', 'catch-up', 'gold star', 'take your hat off to sb', 'piece', 'supporter', 'disadvantage', 
	'first past the post', 'network', 'tremendous', 'gender gap', "dean's list", 'mortarboard', 'stratum', 'age', 'agnomination', 'dream', 
	'encore', 'tread', 'Golden Globe', 'saga', 'raw score', 'a hard (or tough) act to follow', 'unravel', 'Companion of Honour', 'field night', 
	'letter jacket', 'reverent', 'disability', 'inscribe', 'press release', 'graduate', 'title-deed', 'high culture', 'Speranski', 'triple-double', 
	'inscription', 'mark', 'hold', 'lambrequin', 'sinister', 'dubious', 'height', 'Senanayake, Don Stephen', 'hold back', 'impressive', 
	'vital revolution', 'EGOT', 'effect', 'palm', 'honorary degree', 'vegetate', 'letter sweater', 'Oscar', 'tracking', 'Order of Australia', 
	'double-double', 'etrog', 'social promotion', 'timber', 'track', 'baby', 'honours system', 'minimalist', 'halma', 'equanimous', 'liberate', 
	'genius', 'ideal', 'plaque', 'standard test', 'Menshevik', 'attain', 'class system', 'cum laude', 'performance indicator', 'tiger mother', 
	'maturation', 'tribute', 'beachhead', 'cpi', 'grade point average', 'book', 'gee-whiz', 'posttest', 'racism', 'status system', 'authentic assessment', 
	'completion', 'contend', 'magna cum laude', 'scientific socialism', 'acclamation', 'cipher', 'partnership', 'tradition', 'empire builder', 
	'business unionism', 'guerdon', 'Nobel prize', 'standing', 'bay', 'curve', 'intelligence quotient', 'Spingarn, Joel Elias', 'bovarism', 
	'agnomen', 'selection', 'criterion', 'accolade', 'progression', 'culmination', 'hat trick', 'scholastic', 'civilization', 'emphasize', 
	'lifetime', 'self', 'ascetic', 'ascesis', 'Spartan', 'asceticism', 'Protestant ethic', 'Hinayana', 'penance', 'deportment', 'mortify', 
	'self-discipline', 'selfâ€“disciplined', 'self-denial', 'self-mortification', 'disciplinary', 'disciplinarian', 'philosophy', 'will', 'practice', 
	'selves', 'break', 'Bushido', 'Puritan ethic', 'discipline', "Boys' Brigade", 'indiscipline', 'anal character', 'Moism', 'self-control', 
	'school', 'chasten', 'train', 'anthroposophy', 'Jina', 'jnanaâ€“marga', 'disciplined', 'mortification', 'ego', 'exercise', 'consequential', 
	'sell', 'strict', 'undisciplined', 'perfect', 'soi-disant', 'yoga', 'bumptious', 'autogenous', 'autonomy', 'rajaâ€“yoga', 'self-regard', 
	'autogenetic', 'self-government', 'auto-', 'onanism', 'church', 'autolatry', 'amour-propre', 'possessed', 'autogeneal', 'brass', 'self-contained', 
	'egotism', 'disple', 'disciple', 'bundobust', 'self-rule', 'launderette', 'multidiscipline', 'subdiscipline', 'cocky', 'self-command', 
	'sufficient', 'aplomb', 'polarity therapy', 'self-deceit', 'regiment', 'collected', 'herself', 'self-induced', 'flagellate', 'selfism', 
	'egoism', 'hoity-toity', 'chastise', 'harum-scarum', 'inordinate', 'martinet', 'Irish system', 'flagellant', 'form', 'dean', 'autodidact', 
	'self-abuse', 'confidence', 'autarchy', 'denial', 'autogenic', 'self-abnegation', 'self-assured', 'fractious', 'softy', 'indulgent', 
	'address', 'self-love', 'yourself', 'sufficiency', 'himself', 'myself', 'autogamy', 'self-mastery', 'communion', 'mathesis', 'nurture', 
	'methodology', 'truism', 'cockalorum', 'self-taught', 'egotist', 'egotistic', 'schooling', 'penitentiary', 'assured', 'autokinesy', 
	'lick into shape', 'confident', 'obstreperous', 'Zen', 'yogi', 'ungoverned', 'regular', 'wayward', 'automotive', 'self-restraint', 
	'automorphic', 'big', 'autofluorescence', 'flatulent', "on one's feet", 'acquit', 'severe', 'coolness', 'egocentric', 'self-praise', 
	'self-defence', 'oneself', 'phlegm', 'temperance', 'pride', 'tyranny', 'unregulated', 'master-at-arms', 'austere', 'amour propre', 
	'self-esteem', 'smirk', 'autohypnosis', 'unself-conscious', 'pluviograph', 'unself-consciousness', 'unself-consciously', 'brash', 'automath', 
	'sober-minded', 'abstentious', 'indocile', 'orderly', 'manage', 'simper', 'austerity', 'Prussian', 'plant medicine', 'L', 'self-deception', 
	'possession', 'empirical ego', 'reaper-and-binder', 'self-protection', 'abnegation', 'selfâ€“worth', 'self-assurance', 'obstinate', 
	'disciplinatory', 'unruly', 'self-worth', 'pompous', 'inwardly', 'control', 'selfâ€“pollinate', 'stuffy', 'autopoiesis', 'autoinduction', 
	'autonomous', 'autotheism', 'scleragogy', 'disciplining', 'boisterous', 'rank', 'licentious', 'discipliner', 'pedagogy', 'trainer', 
	'disciplinarily', 'nibs', 'willy', 'self-admiration', 'decadent', 'self-made', 'self-pitying', 'uncreate', 'portentous', 'upstart', 
	'aut-', 'self-sterile', 'impotent', 'selfâ€“aware', 'destruct', 'lonesome', 'sen', 'SASE', 'prima facie', 'adjoin', '-teria', 'willing', 
	'assurance', 'timid', 'self-determining', 'commune', 'masturbation', 'closed', 'self-conscious', 'ultramontane', 'scholarship', 'self-respecting', 
	'smug', 'drill', 'swaraj', 'herd', 'council', 'flagellation', 'go', 'decadence', 'temperate', 'whip', 'sadhana', 'uncool', 'biggity', 'barograph', 
	'self-harm', 'groceteria', 'diffident', 'automatic', 'conscious', 'autarky', 'sovereignty', 'vaunt', 'alter ego', 'Meta', 'self-satisfied', 
	'self-imposed', 'self-destruction', 'self-confidence', 'perk', 'person', 'self-fertile', 'self-abasing', 'jack-in-office', 'proud', 'poised', 
	'selfâ€“satisfied', 'itself', 'self-centered', 'autotheist', 'egomania', 'singly', 'forbearance', 'self-rising flour', 'absenter', 
	'self-aggrandizement', 'dogs', 'Peacock', 'hubris', 'thermograph', 'enforce', 'self-flagellation', 'a firm hand', 'demoralize', 'history', 
	'disciplinant', 'hard labor', 'master', 'covenant', 'mulct', 'drillmaster', 'rod', 'breed', 'ferule', 'undisciplinable', 'fundamentalist', 
	'praxis', 'furnace', 'wild', 'undiscipline', 'tutor', 'corrective', 'bioethics', 'behave', 'Curry', 'disciplinal', 'self-propagating', 
	'independent', 'sycophant', 'continent', 'Bull', 'let', 'perverse', 'subdue', 'tame', 'targe', 'pratimoksha', 'contrary', 'inurement', 
	'arrest', 'autometry', 'continency', 'disciplinableness', 'docile', 'penitential', 'punishment', 'transdisciplinary', 'uncorrected', 
	'incorrection', 'uncontrolled', 'rigor', 'studied', 'relaxation', 'selfâ€“contented', 'selfâ€“contentment', 'selfâ€“devoting', 'havlagah', 
	'selfgratulatory', 'self-sustenance', 'self-annihilation', 'selfâ€“loved', 'selfâ€“opinionative', 'selfâ€“picture', 'self-directed', 
	'selfâ€“betrayal', 'selfâ€“pleased', 'selfâ€“professed', 'integral cover', 'egoâ€“satisfaction', 'trivial', 'carpenter weed', 'selfâ€“collected', 
	'selfâ€“elected', 'selfâ€“involved', 'selfâ€“reproducing', 'logical', 'autofecundation', 'selfâ€“styled', 'rumbustious', 'restraint', 'slap',
	'ruly', 'jnanaâ€“yoga', 'methodological', 'father', 'study', 'official', 'run wild', 'disarray', 'disciplinable', 'ramrod', 'breeding', 
	'rambunctious', 'robustious', 'selfâ€“gratulation', 'selfâ€“exam', 'autoregulation', 'selfâ€“endeared', 'selfâ€“consequence', 'selfâ€“confrontation',
	'selfâ€“scrutiny', 'selfâ€“relation', 'selfâ€“conception', 'selfâ€“seed', 'selfâ€“subsisting', 'selfâ€“movement', 'selfâ€“caused', 'selfâ€“assumption',
	'transcendental self', 'self-gratification', 'selfâ€“naughting', 'selfâ€“trust', 'selfâ€“respectful', 'hookweed', 'selve', 'self-repulsive', 
	'selfâ€“impotent', 'autokinetical', 'selfâ€“ruling', 'automobile', 'self-devotement', 'self-charity', 'selfâ€“government', 'idiolatry', 'autodidacticism', 
	'selfâ€“involvement', 'butt', 'selfâ€“admiration', 'self-understanding', 'selfâ€“forgetting', 'autoctony', 'autoregulating', 'selfâ€“fertilizing', 'selfâ€“revelative',
	'self-limiting', 'selfâ€“rule', 'auto-destruct', 'selfâ€“assurance', 'selfâ€“corrective', 'selfâ€“searching', 'selfâ€“deceptive', 'selfâ€“content', 'selfâ€“destroying',
	'self-commune', 'selfâ€“involution', 'blue lucy', 'self-sustaining', 'selfâ€“limiting', 'selfâ€“sufficing', 'selfâ€“opinioned', 'autophagous', 'autoregulated', 'self-repairing',
	'bootstrap', 'meta-', 'self-dependent', 'impotence', 'impotency', 'self-denying', 'self-assertive', 'self-righteous', 'self-abandonment', 'abstinence', 'self-imposture', 'self-enjoyment',
	'masturbatory', 'self-stick', 'inconsistent', 'self-trust', 'nonscience', 'hermit', 'unlearned', 'harden', 'mereology', 'chapter', 'punish', 'autolysis', 'BSE', 'logic', 'blast', 'prefect',
	'ungirt', 'light', 'lightsâ€“out', 'geomatics', 'karma yoga', 'gender studies', 'mp', 'marshal', 'pest', 'classical', 'folklore', 'hell ship', 'loosen', 'immortification', 'masterâ€“atâ€“arms',
	'rigid', 'woodshed', 'lawless', 'unmanageable', 'disciplinize', 'raciology', 'perversity', 'museology', 'religious studies', 'coltish', 'disciplinated', 'enregiment', 'recalcitrant', 'signific',
	'routinize', 'silent system', 'sociologism', 'relax', 'selfâ€“flattering', 'selfâ€“feeling', 'warm', 'Se', 'small self-administered pension scheme', "throw one's weight about (or around)", 
	'self-willed', 'self-pity', 'paradoxical', 'monastic', 'micromania', 'jaunty', 'heal-all', 'distemper', 'disinterested', 'complacent', 'braggart', 'amok', 'automatic door', 'axiomatic', 'attitude',
	'self-ignorant', 'healâ€“all', 'incontinence', 'speaking rod', 'selfâ€“propelling', 'selfâ€“motive', 'selfâ€“congratulatory', 'masturbate', 'locomotion', 'increate', 'flatulence', 'SPM', 
	'self-reflection', 'self-consistent', 'oleaginousness', 'unselfconscious', 'diffidence', 'debauched', 'easy', 'SAE', 'selfâ€“revealing', 'selfâ€“reflection', 'selfâ€“life', 'selfâ€“governance',
	'selfâ€“esteem', 'selfâ€“denying', 'selfâ€“applauding', 'recover', 'ownself', 'personally', 'SLR', 'unit', 'surety', 'pontifical', 'overblown', "lose one's head", 'hercogamous', 'free', 'carry', 
	'axiom', 'self-repugnant', 'self-gratulation', 'self-delation', 'self-centring', 'self-applause', 'self-annihilated', 'self-abhorrence', 'unbegotten', 'cock-a-hoop', 'travel', 'boastful', 'bragging',
	'objective', 'free-living', 'dignity', 'cafeteria', 'SIW', 'egotheism', 'increated', 'greatness', 'toplofty', 'notself', 'self-absorbed', 'self-depreciation', 'self-doubt', 'incontinent', 'shy', 
	'unity', 'heautophany', 'self-seed', 'self-depending', 'SRO', 'pushful', 'self-reproaching', 'bounce', 'soppy', 'sorry for oneself', 'shit-eating', 'causeless', 'non-', 'selfâ€“doubting', 
	'selfâ€“governed', 'selfâ€“murder', 'selfâ€“restraining', 'selfâ€“serve', 'rectitudinous', 'bashful', 'collect', 'selfâ€“seeker', 'selfness', 'grip', 'locomotive', 'inner space', 'self-centering',
	'self-devised', 'anemograph', "be one's own boss", 'aw-shucks', 'autoload', 'being', 'pushing', 'ruth', 'rÃ©clame', 'self-', 'restrained', 'selfâ€“preservative', 'front', 'selfâ€“affected',
	'selfâ€“cocker', 'selfâ€“knowing', 'betake', 'autokinetic', 'locknut', 'effete', 'pamper', 'imprudent', 'sentient', 'self-contempt', 'self-criticism', 'self-fulfilling', 'self-justification',
	'self-torture', 'selfie', 'unsure', 'benign', 'obey', 'allheal', 'auto-portrait', 'masquerade', 'primitive', 'vainglorious', 'selfâ€“confiding', 'purdy', 'outrecuidance', 'self-concern',
	'philauty', 'by-interest', 'unction', 'nainsel', 'self-supporting', 'assertive', 'sang-froid', 'municipal', 'selfâ€“repugnant', 'hair shirt', 'three-hanky', 'tale', 'alone', 'dumb', 'washateria', 
	'soiâ€“disant', 'sober', 'paddle', 'oat', 'motor', 'conscience', 'selfâ€“identical', 'masterful', 'dedication', 'selfâ€“denied', 'apperception', 'headstrong', 'intradisciplinary', 'expedient',
	'poise', 'self-estimation', 'self-activity', 'selfâ€“actor', 'conceit', 'self-devouring', 'booketeria', 'self-sealing', 'Mushet steel', 'selfâ€“poised', 'selfâ€“sacrificer', 'heartâ€“searching',
	'kamikaze', 'jackeen', 'self-contradictory', 'snap', 'tin god', 'autotoxaemia', 'fare', 'udomograph', 'defenceless', 'gut', 'lave', 'lucrative', 'pharisee', 'furnish', 'self-communion', 
	'interested', 'report', 'dogâ€“eatâ€“dog', 'cutesy', 'inmost', 'crack', 'selfâ€“discrepant', 'selfâ€“deceiver', 'selfâ€“appointed', 'dignified', 'sel', 'self-forgetful', 'ruthful', 'pushy',
	'self-incompatible', 'pleasure', 'selfâ€“realizationist', 'stay', 'clasp lock', 'soul', 'autogamous', 'donnism', 'self-reprovingly', 'dandified', 'disinvoltura', 'self-delusion', 'autodynamic', 
	'self-worship', 'self-interested', 'autocracy', 'autognosis', 'obscure', 'perky', 'dog', 'fleshpots', 'selfâ€“born', 'consistent', 'bouncy', 'ombrograph', 'selfâ€“denier', 'selfâ€“incompatible',
	'fanfaronade', 'complacency', 'truculent', 'dirigible', 'asocial', 'self-serving', 'self-approbation', 'selfish', 'self-regulating', 'plain', 'self-indulgent', 'face', 'innermost', 'navelâ€“gazing',
	'heteropathic', 'self-consciously', 'sit', 'peevish', 'humiliation', 'knuckle', 'consciousness', 'discover', 'dramatize', 'native', 'bug', 'strut', 'self-parodic', 'self-evaluation', 'self-depreciatory',
	'self-explanatory', 'barometrograph', 'bullish', 'self-satisfying', 'seity', 'phlegmatic', 'ESS', 'look', 'crow', 'intermeddle', 'kip', 'luxurious', 'maintain', 'smugly', 'self-compatible', 'self-slaughter',
	'feisty', 'freedom', 'dogged', 'embarrass', 'holy', 'hubristic', 'hyetometrograph', 'superfly', 'marigraph', 'selfâ€“asserting', 'selfâ€“assured', 'selfâ€“important', 'crouse', 'self-consuming', 'bike',
	'cheap', 'composed', "drop one's bundle", 'crust', 'chutzpah', 'assure', 'atman', 'airship', 'abasement', 'insecure', 'self-inflicted', 'self-sacrifice', 'catechize', 'rule', 'martinetism', 'keelhaul', 
	'neurosurgery', 'BDSM', 'musar', 'infonomics', 'rigorous', 'training', 'hebdomader', 'agrology', 'interdisciplinarity', 'heel', 'managerialism', 'nomenclature', 'fundamentalism', 'slopestyle', 'terminology',
	'military psychology', 'social studies', 'technician', 'languaged', 'golden rule', 'chicken', 'boot camp', 'disfellowship', 'loose', 'disciplination', 'timeâ€“out', 'engineering', 'fast', 'family values',
	'educate', 'technology', 'sociogenomics', 'sanction', 'lick', 'cultivation', 'chastisement', 'basic training', 'chickenshit', 'brief', 'metaphysics', 'spit and polish', 'taskmaster', 'Ritsu', 'bystander', 
	'censure', 'correct', 'language', 'like', 'pentathlon', 'permissive', 'retribution', 'semasiology', 'sternly', 'method', 'catechumen', 'exact', 'churchism', 'prussianize', 'architecture', 'prudence', 'paradigm',
	'cultural', 'historiography', 'reformatory', 'revolution', 'science', 'tariqa', 'secret society', 'terminal degree', 'asËˆcetiËŒcism', 'confine', 'Makarov', "in one's own right", 'shrinking violet', 'sordid',
	'selfâ€“analytical', 'selfless', 'selfâ€“reflective', 'egoâ€“expansion', 'hedonism', 'individualism', 'self-seeking', 'intrapersonal', 'disinterest', 'motorcar', 'assuming', 'absquatulate', 'hurl', 
	'self-distrust', 'anima', 'reseed', 'gingerness', 'warefulness', 'wariness']


#Extraversion
extra= ['bar', 'other', 'drinks', 'restaurant', 'dancing ', 'restaurants ', 'cats', 'grandfather  ', 'Miami  ', 'countless  ', 'drinking  ', 'shots  ', 
'computer  ', 'girls  ', 'glorious ', 'minor ', 'pool  ', 'crowd ', 'sang  ', 'grilled', 'house', 'clothes', 'money', 'dining', 'travel', 'car', 'drink', 
'table', 'clothes', 'bottle', 'trip', 'lunch', 'father', 'mother', 'brother', 'sister', 'singing', 'swimming', 'laptop', 'talkative', 'social', 'congenial', 'gregarious', 'personable', 'sociable', 'cordial', 'demonstrative', 'friendly', 'adaptability', 'companionship', 'boldness ', 'brashness', ' forwardness ', 'immodesty', 'camaraderie', 'companionship', 'fellowship', 'amiability', 'cordiality ', 'folksiness', 'friendliness', 'neighborliness', 'conviviality', 'gregariousnessy', 'Friends ', 'Leisure ', '1st Person Pl. ', ' Family ', 'Other Refs. ', ' Up ', 'Social Processes ', ' Positive Emotions  ', 'Sexual  ', 'Space ', 'Physical States ', ' Home ', 'Sports ', 'Motion  ', 'Music ', 'Inclusive ', 'Eating ', 'Time ', 'Optimism ', 'Causation', 'Music ', 'Positive Feelings ', 'Affect  ', 'Friends ', 'sexual', '2nd Person', 'Leisure  ', 'Physical States ', 'Assent ', '1st Person Pl', 'Other Refs. ', 'Total Pronouns ', 'Eating ', 'Seeing ', 'Social Processes ', 'Space ', 'Motion ', 'Body States', 'Person', 'Checking ', 'excitement', 'love  ', 'kidding ', ' hot ', ' friends ', 'spend  ', 'shots', 'glory ', ' mss ', 'sing ', 'girls  ', 'perfect ', ' denied ', ' sweet', 'song  ', 'every  ', 'temporary ', ' dance  ', 'golden ', 'Openness\t', 'sang', 'hotel ', 'lazy ', 'kissed', 'shots ', 'golden ', 'dad', 'girls ', 'restaurant ', 'eve', 'best ', 'proud', 'accept ', 'soccer', 'met', 'not', 'brothers', 'interest', 'cheers','bonhomie', 'amicable', 'warmth', 'affable', 'fellowship', 'gemï¿½tlichkeit [or] gemutlichkeit', 'friendship', 'gemÃ¼tlichkeit', 'thaw', 'sociality', 'sociable', 'friending', 'greeting', 'friendling', 'environment-friendliness', 'genial', 'cordial', 'cool', 'love feast', 'coolness', 'geniality', 'wink', 'friendlihood', 'goodwill', 'amiable', 'friendly', 'repulsive', 'chum', 'estrange', 'empressement', 'hail-fellow-well-met', 'unapproachable', 'hearty', 'at peace', 'stony', 'amicability', 'familiarity', 'welcome', 'good will', 'community spirit', 'biofriendly', 'distance', 'companiable', 'treat', 'unfriendly', 'glacial', 'amity', 'cold', 'mistake', 'ice', 'intimate', 'veneer', 'kindliness', 'reconcile', 'smooth-faced', 'coolly', 'conviviality', 'warm', 'regard', 'comity', 'alien', 'handshaker', 'alienate', 'sympathetic', 'affability', 'marblehearted', 'camaraderie', 'jovial', 'brittle', 'disaffect', 'unchanging', 'greenwash', 'charm offensive', 'noticeable', 'unctuous', 'personality', 'normalize', 'fawn', 'smile', 'courteous', 'abatement', 'lovable', 'clubby', 'subderivative', 'cordially', 'hospitality', 'congeniality', 'synthetic', 'loving', 'gamble', 'gene', 'go-ahead', 'macho', 'machismo', 'spine', 'forwardness', 'ipsedixitism', 'crust', 'bluster', 'spirit', 'unaggressive', 'belligerent', 'positivity', 'positivism', 'get-tough', 'shyness', 'adventurous', 'thrill', 'delirium', 'exciting', 'fever', 'the thrill of the chase', 'electricity', 'sensation', 'orgasm', 'flutter', 'self-seeking', 'intoxication', 'stir', 'heat', 'calm', 'fermentation', 'frenzy', 'twitter', 'inquiry', 'rapture', 'buzz', 'breathless', 'purchase', 'flurry', 'commotion', 'ferment', 'up', 'delirious', 'dither', 'gallivant', 'suspense', 'humdrum', 'eroticism', 'feverish', 'courtship', 'hedonism', 'spice', 'fire', 'ablaze', 'fever pitch', 'excited', 'kick', 'excitation', 'demand', 'hunt', 'rousing', 'furore', 'frisson', 'madness', 'valetudinarian', 'enthusiasm', 'retrieve', 'glamorous', 'agitation', 'quest', 'romance', 'hair-raising', 'fuss', 'zetetic', 'squee', 'cool', 'stew', 'erogenous', 'sensational', 'hectic', 'thrilling', 'flush', 'hullabaloo', 'tizzy', 'tremor', 'aspirant', 'tingle', 'quiet', 'breathtaking', 'excitatory', 'search', 'rut', 'whoop', 'glamour', 'desirable', 'tame', 'ebullient', 'aflame', 'rackety', 'suitor', 'erethism', 'animated', 'dance', 'hyperexcitement', 'premotion', 'atingle', 'sycophant', 'jazz', 'mania', 'furor', 'uproar', 'flap', 'hysteria', 'lather', 'hurrah', 'wired', 'effervesce', 'expediential', 'vengeful', 'lily-white', 'delicious', 'sputter', 'red-hot', 'letdown', 'rock', 'simmer down', 'gadabout', 'bang', 'zest', 'high', 'sizzle', 'tumult', 'judicial', 'carpetbagger', 'exhilaration', 'seethe', 'hoopla', 'roar', 'prosaic', 'exuberant', 'frantic', 'hot', 'evasive', 'find', 'sedative', 'flat', 'erotic', 'bland', 'alarm', 'arrah', 'wild', 'fluster', 'simmer', 'excitement', 'seeker', 'suit', 'combustion', 'stirring', 'fast', 'blimey', 'tremble', 'tension', 'bubble', 'anticipation', 'tickle', 'nightlife', 'pornography', 'reclusive', 'salt', 'fry', 'shiver', 'crazy', 'coprophilia', 'nonchalant', 'nervine', 'hubbub', 'buck fever', 'enthuse', 'gale', 'gaiety', 'fund-raising [or] fundraising', 'excitable', 'on the warpath', 'hedonist', 'El Dorado', 'asocial', 'dead', 'concurrent', 'refugee', 'carpetbag', 'glamorize', 'cooler', 'whee', 'enthusiastic', 'fast-paced', 'fury', 'racket', 'key up', 'white heat', 'warm', 'uninspired', 'agog', 'irritation', 'ardor', 'charged', 'taking', 'unglamorous', 'manic-depressive', 'downbeat', 'ecstasy', 'blaze', 'atonic', 'knocker', 'descriptive', 'boy', 'self-serving', 'hooray', 'brinkmanship [or] brinksmanship', 'solicitation', 'fuel', 'worldling', 'biopsy', 'hairâ€“raising', 'overexcitement', 'inflammation', 'fume', 'flame', 'rapturous', 'hoo-ha', 'lifeless', 'acquisitive', 'titillate', 'lenitive', 'persuasion', 'selfish', 'thump', 'steady', 'glee', 'opportunism', 'ripsnorter', 'hot dog', 'inexcitable', 'pizzazz', 'spring fever', 'eureka', 'vibrant', 'still', 'slaver', 'blow-off', 'elation', 'tranquil', 'yes', 'olï¿½ [or] ole', 'feeze', 'impatience', 'rogatory', 'look', 'alight', 'erect', 'postulant', 'colour', 'gregarious', 'sexy', 'geeâ€“whiz', 'jerk', 'euphoria', 'rhapsody', 'sedation', 'glowing', 'stimulation', 'ree', 'fizz', 'rah-rah', 'wow', 'electrify', 'aflare', 'jump', 'lively', 'after', 'asylee', 'fundraising', 'surf', 'Pan', 'nocturnal', 'political', 'eh', 'cougar', 'dating agency', 'selfâ€“justifying', 'exhibitionist', 'scout', 'jobseeker', 'ravenous', 'developing country', 'spew', 'crackle', 'gold fever', 'manic', 'hot-blooded', 'wishful', 'upseeking', 'banzai', 'selfâ€“seeker', 'brouhaha', 'intensity', 'bustle', 'pheese', 'depreciation', 'disquisition', 'what', 'sedate', 'wind up', 'Low', 'mad', 'spasmodic', 'zap', 'glow', 'holy', 'berserk', 'beside', 'booming', 'all right', 'turmoil', 'driving', 'headâ€“hunting', 'hwyl', 'fleshment', 'opposition', 'heat-seeking', 'research', 'titillating', 'liberation', 'lucrative', 'withdrawn', 'politicking', 'pop-eyed', 'apathy', 'adventure', 'employment agency', 'yippee', 'heady', 'ride', 'maniacal', 'fast-living', 'to-do', 'fish', 'joie de vivre', 'underground', 'faction', 'titillation', 'dampen', 'circus', 'singles bar', 'dramatic', 'recluse', 'deadly', 'hound', 'seekingly', 'town', 'breathlessly', 'escapade', 'serenely', 'razzle-dazzle', 'allure', 'adither', 'susceptibility', 'gog', 'lackluster', 'aglow', 'astonishment', 'hoigh', 'erotogenic', 'larry', 'my', 'ginger', 'inflame', 'dull', 'racketry', 'nervously', 'pizazz', 'amusement', 'ennui', 'pride', 'apoplexy', 'salutation', 'drama', 'irritate', 'rage', 'frantically', 'rhapsodist', 'moderate', 'blow', 'temperature', 'yarn', 'deprecatory', 'sordid', 'forage', 'church errant', 'pulse', 'jacked', 'hedonic', 'whirlpool', 'kerfuffle', 'savour', 'selfâ€“seeking', 'recourse', 'pleasurable', 'luxurious', 'pleasuremonger', 'sortes', 'rent-seeker', 'goose bumps', 'candidate', 'arriviste', 'emulous', 'climber', 'extroitive', 'inflammatory', 'rampage', 'blasÃ©', 'minimalist', 'get a bang out of', 'uninspiring', 'unanimated', 'verve', 'thrillful', 'high-spirited', 'obscene', 'giddy-making', 'horniness', 'tantalizing', 'raving', 'nervism', 'antiseptic', 'intoxicated', 'erotism', 'doolally', 'fidgin fain', 'provocative', 'dullness', 'action-packed', 'pandemonium', 'abubble', 'pallid', 'jazzed up', 'horny', 'geeked', 'suffragette', 'devastavit', 'job counselling', 'out', 'appetition', 'cosy', 'stroll', 'unflashy', 'unworldly', 'positive pole', 'ask', 'Chicano movement', 'celebutante', 'introduction agency', 'etiologic', 'fling', 'South Pole', 'be gunning for', 'browse', 'unemployed', 'make', 'humanitarian', 'one-man', 'party', 'discouraged worker', 'evangelistic', 'sabre-rattling', 'subversive', 'scent', 'mediator', 'adventuresome', 'antimissile', 'money-grubbing', 'social', 'Sinn Fein', 'temperance movement', 'osmosis', 'hastily', 'raptus', 'shudder', 'barn burner', 'weariness', 'ampedâ€“up', 'easy', 'peaceful', 'groove', 'doodah', 'clinical', 'boil', 'arid', 'arouse', 'hot diggity', 'excite', 'phrenetic', 'abuzz', 'frothy', 'hooâ€“ha', 'monochromatic', 'apatheia', 'heart-stopping', 'weak-kneed', 'coolness', 'dryness', 'orgasmic', 'yow', 'effervescence', 'airless', 'combust', 'disorder', 'ole', 'sparkly', 'transports of joy/delight/rapture', 'send', 'yahoo', 'swither', 'unrestrained', 'in (or of) a twitter', 'fervor', 'faunch', 'all of a flutter', 'stodgy', 'twitterpated', 'animation', 'Jove', 'hoâ€“hum', 'feverishly', 'hurray', 'rip-roaring', 'rowdydow', 'rouse', 'state', 'hubbaâ€“hubba', 'spark', 'unconfined', 'live', 'highly charged', 'pumped', 'effervescent', 'excitedly', 'stoked', 'sent', 'throb', 'fantigue', 'unenthusiastic', 'turn-on', 'exhilarating', 'hot up', 'carryâ€“on', 'set the world/place etc alight', 'stitherum', 'bright lights', 'grab', 'infectious', 'crescendo', 'liveliness', 'hysterics', 'phrensy', 'scintillating', 'take the shine off', 'repose', 'heavy', 'aflutter', 'be bouncing off the walls', 'amped', 'boil over', 'bleak', 'cor', 'placid', 'aseptic', 'yay', 'commonplace', 'munch', 'white-knuckle', 'emotion', 'avaricious', 'hitchhike', 'breeding-ground', 'browser', 'seen', 'innit', 'sexually harassing', 'voyeur', 'hypergamy', 'histrionic', 'probe', 'Hindutva', 'overreach', 'redress', 'funny', 'underbid', 'offensive', 'revanchism', 'sign off', 'special interest group', 'high profile', 'shortâ€“short', 'succorance', 'abraham-man', 'counter-revolution', 'guidance', 'schism', 'status-seeking', 'berrying', 'widow-hunter', 'divination', 'doctrinaire', 'dangle', 'embusquÃ©', 'hegemonistic', 'gate net', 'language police', 'guided missile', 'extreme', 'troll', 'earnest', 'renowning', 'sign', 'tremulous', 'warmth', 'heart', 'romantic', 'tumultuous', 'enragement', 'ery', 'adrenaline junkie', 'build-up', 'hypergetic', 'imagination', 'pacy', 'euphoric', 'frenzied', 'sensationally', 'sterile', 'make a song and dance about (something)', 'pop eye', 'burning', 'gay', 'whip', 'whoo', 'air', 'drunkenness', 'sizzling', 'agony', 'ambulance chaser', 'statehood', 'hamesucken', 'personal', 'tattletale', 'help wanted', 'lobby', 'wanderjahr', 'rise', 'searching', 'twitch', 'unsocial', 'something', 'parrhesia', 'floodgate', 'revenge', 'rescue mission', 'politics', 'status symbolism', 'consultation', 'countersign', 'consciousness-raising', 'demented', 'ecstatic', 'spine-chiller', 'widdrim', 'wanton', 'worldâ€“weary', 'zip', 'vibrate', 'aquiver', 'soulless', 'music', 'dynamite', 'gad', 'lukewarm', 'mayhem', 'inirritative', 'shake', 'wind down', 'apathetic', 'rhapsodize', 'carry on', 'febrile', 'gleam', 'a storm in a teacup', 'into orbit', 'hot damn', 'electric', 'novelty', 'life', 'jimjams', 'intoxicating', 'big deal', 'go mad', 'antiorgastic', 'drab', 'exhilarate', '(Oh) boy!', 'palaver', 'awesome', 'tempestuous', 'frenetic', 'sharking', 'brain', 'dating service', 'cute', 'descriptory', 'casual ward', 'entitle', 'pathfinder', 'shotgun', 'tinker', 'restorationism', 'pussy posse', 'lonely hearts', 'peacemonger', 'bankrupt', 'tenacious', 'animalculism', 'gyrocompass', 'rousement', 'underwhelmed', 'madhouse', 'hared up', 'Shield fever', 'exultation', 'uninteresting', 'blush', 'hyperventilate', 'spine-chilling', 'holy cow', 'hangover', 'bristle', 'bounce', 'calmness', 'dare-devil', 'inexcitability', 'bingo', 'passionate', 'toss', 'lift', 'toe-curling', 'melodramatic', "get one's jollies", 'energize', 'hubba-hubba', 'underwhelming', 'tew', 'excitedness', 'suggestive', 'alt', 'lark', 'berko', 'snorter', 'rush', 'astir', 'breathe (new) life into', 'alarums and excursions', 'touchpaper', 'soliitation', 'feaze', 'beside oneself', 'communism', 'enemy', 'escapism', 'magnetic north', 're-reformation', 'gradualism', 'tufthunting', 'accident investigation', 'kettle', 'prior', 'a bird in the hand is worth two in the bush', 'fehme', 'petal', 'job club', 'knock', 'second papers', 'lawsuit', 'thoughtful', 'casting couch', 'escapist', 'junket', 'junketeer', 'jindyworobak', 'rhubarb', 'radical', 'separatist', 'special interest', 'status-dissenting', 'ear-tickling', 'business', 'asylum seeker', 'public relations', 'the struggle for existence (or life)', 'Fabian', 'derogation', 'trial', 'woo', 'dienophile', 'edge', 'feeding frenzy', 'gulp', 'hotness', 'nervous', 'sang-froid', 'ZOMG', 'aburn', 'twittering', 'wave', 'auto-erotic', 'come', 'come down/back to earth', 'flagellation', 'butterfly', 'brainstorm', 'depressant', 'ooh', 'swashbuckler', 'raise', 'racquet', 'roaring', 'thrill kill', 'yell', "sb's heart/mind/pulse races", 'slow', 'spice sth up', 'charge', 'freak-out', 'fly', 'fetish', 'undue', 'sadism', 'foam at the mouth', 'jolly', 'jumpy', 'sensationalism', 'rave', 'fantod', 'tempest in a teapot', 'tantalize', 'erection', 'pop', 'bloom', 'boiling point', 'interest', 'storm', 'sleepy', 'passion', 'race', 'feel sb up', 'hotting-up', 'hair', 'exuberance', 'fevered', 'speak in tongues', 'go mental', 'trembling', 'light up', 'nonchalance', 'priapic', 'spice up something', 'squirm', 'the calm/lull before the storm', "someone's heart is knocking", 'hell', 'irritable', 'goosebumps', 'exies', 'maenad', 'owly-eyed', 'false alarm', 'saturnine', 'flare', 'froth at the mouth', 'hubba hubba', 'holy roller', 'exclamation', 'light', 'in full cry', 'debilitant', 'a blaze of publicity/glory', 'glitter', 'low-key', 'pant', 'come (or bring someone) back (down) to earth', 'exult', 'shivery', 'cyclothymia', 'wow factor', 'orgiastic', 'hum', 'opa', 'atremble', 'generate', 'get', 'spike', 'let off/blow off steam', 'brandish', 'amphetamine', "lick one's chops", 'drunk', 'horripilation', 'screamer', 'resort', 'recovery movement', 'spiritual director', 'investigative', 'agony column', 'coping', 'exclaim', 'come to life', 'glue', 'life in the fast lane', 'lyrical', 'prester', 'bound', 'thrills and spills', 'amove', 'get your kicks from something', 'coolly', 'surfer', 'phallus', 'whip into', 'breeze', "lick one's lips", 'set (something) on its ear', "the pit of one's/the stomach", 'gambler', 'subdue', 'swinger', 'woot', 'suburban', 'set the world on fire', 'someoneâ€™s stomach churns/lurches/tightens', 'fey', 'contain', 'autoeroticism', 'ascetic', 'caucus', 'pap', 'evangelical', 'swagman', 'Tammany', 'benefit tourist', 'sundowner', 'chase', 'Jobclub', 'kerb crawling', 'Mahayana', 'reality principle', 'mavericker', 'strike', 'monastery', 'canvass', 'take', 'anthroposophy', 'expressionist', 'fang shih', 'hot money', 'junk bond', 'forum shopping', 'employment service', 'box trap', 'hedonistic', 'opportunity school', 'go', 'wanted circular', 'consumerism', 'employment agent', 'girling', 'business union', 'gunslinger', 'new woman', 'hard goer', 'charismatic', 'expressionism', 'kairomone', 'speed dating', 'vulture', 'letters of request', 'duty', 'plummet', 'challenge', 'crambo', 'loser', 'gestalt psychology', 'criminal investigation', 'action', 'direct action', 'MOB', 'peaceable', 'thriller', 'scamper', 'pump up', "be a nine days' wonder", 'blood sport', 'jolly hockey sticks', 'haste', 'pacify', 'on tenterhooks', 'must', 'hokum', 'channel fever', 'buoyancy', 'cheery', 'sunshine', 'lightness', 'cheer', 'radiance', 'spiritless', 'hilarity', 'grim', 'cheerisness', 'pleasantry', 'merry', 'morale', 'gladsome', 'geniality', 'exuberant', 'Blood', 'heyday', 'lift', 'brightness', 'livelihood', 'comfortably', 'glad', 'radiate', 'cheerful', 'mirth', 'allegresse', 'heartening', 'the gaiety of nations', 'bright', 'cordial', 'upbeat', 'cardiac', 'joy', 'jollity', 'sanguine', 'genialness', 'infestivity', 'chipper', 'gladness', 'riancy', 'gay', 'genialize', 'exuberance', 'saturnine', 'gaiety', 'vivacity', 'relief', 'unpleasantry', 'cheerily', 'genial', 'spirit', 'happy pill', 'native', 'perk', 'jovial', 'heavy', 'light', 'grace']


op= calcy(tokens,openn)
openness= (op/lengy)
openness=round(openness,2)

c= calcy(tokens,consc)
consciousness= (c/lengy)
consciousness=round(consciousness,2)

e= calcy(tokens,extra)
extraversion= (e/lengy)
extraversion=round(extraversion,2)

a= calcy(tokens,agree)
agreeableness =(a/lengy)
agreeableness=round(agreeableness,2)

n= calcy(tokens,neuro)
neuroticism= (n/lengy)
neuroticism=round(neuroticism,2)


per_name="report/"+my_name+"/Personality-"+my_name+".png"
data = {'Openness':openness, 'Conscientious':consciousness, 'Extraversion':extraversion,'Agreeableness':agreeableness, 'Neuroticism':neuroticism} 
per = list(data.keys()) 
score = list(data.values()) 
   
fig = plt.figure() 
  
# creating the bar plot 
plt.bar(per, score, color ='#16c79a',width = 0.6) 

plt.xlabel("Personality Traits") 
plt.title("Persona Analysis") 
plt.savefig(per_name)
plt.show()