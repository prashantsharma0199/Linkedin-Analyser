from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import matplotlib.pyplot as plt
from textblob import TextBlob 
import os
import time
import string
import pandas as pd
from nltk.corpus import stopwords

# my_name = input("Enter your name: ")
# f_name = input("Enter the name of your data file you need to analyse (without file extension): ")

# print("Please wait your report is generating.....")

def analysis(my_name, f_name):
	analyzer = SentimentIntensityAnalyzer()

	folder="static/report/"+my_name
	if not os.path.exists(folder):
		os.makedirs(folder)


	time.sleep(5)
	data=open("static/report/"+my_name+"/Report-"+my_name+".txt","w+",encoding="utf-8")


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

		# print("sentence was rated as ", vs['neg']*100, "% Negative")
		# print("sentence was rated as ", vs['neu']*100, "% Neutral") 
		# print("sentence was rated as ", vs['pos']*100, "% Positive")

		if vs['compound'] >= 0.05 : 
			# print("Positive")
			pos=pos+1 
	
		elif vs['compound'] <= - 0.05 : 
			# print("Negative")
			neg=neg+1
	
		else : 
			# print("Neutral")
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

	data.write("Results: " + str_dictionary + "\n\n\n")

	#report generated
	data.close()

	# print(polarity)
	# print("\n")
	# print(subjectivity)

	scat_name="static/report/"+my_name+"/Scatter-"+my_name+".png"
	pie_name="static/report/"+my_name+"/Pie-"+my_name+".png"

	plt.grid()
	plt.scatter(polarity, subjectivity)
	plt.title("Sentimental Analysis")
	plt.xlabel("Polarity")
	plt.ylabel("Subjectivity")
	plt.savefig(scat_name)
	plt.close()
	# plt.show()


	labels = ['Neutral', 'Positive', 'Negative']
	sizes = [neu_per,pos_per,neg_per] 
	explode = (0, 0, 0)  
	fig1, ax1 = plt.subplots()
	ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

	ax1.axis('equal')  
	plt.tight_layout()
	plt.savefig(pie_name)
	plt.close(fig1)
	# plt.show()


	print("Your report has been generated !!")


	#Complete ocean_analysis


	file = open("static/report/"+my_name+"/Report-"+my_name+".txt",'r',encoding='utf-8')


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


	#245 words per category
	#reference used: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2885844/

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
		'temper', 'furious', 'incense', 'glower']
		

	#Openness
	openn=['novel', 'fame', 'urge', 'decades', 'urban', 'glance', 'length', 'poetry', 'literature', 'audience', 'anniversary', 
		'loves', 'narrative', 'lines', 'bears', 'thank', 'humans', 'beauty', 'moon', 'blues', 'sky', 'plants', 'dance', 'beautiful', 
		'trees', 'planted', 'flowers', 'sang', 'blue', 'sings', 'danced', 'music', 'afterwards', 'tree', 'painted', 'hills', 'outdoor', 
		'feel', 'breathe', 'feeling', 'awful', 'stressful', 'stress', 'fabulous', 'felt', 'heart', 'lucky', 'cried', 'overwhelming', 
		'sleep', 'hours', 'scared', 'sick', 'therapy', 'am', 'myself', 'feels', 'streets', 'city', 'century', 'sexual', 'industry', 
		'businesses ', 'south ', 'tour ', 'sean', 'global', 'diaper', 'immigration ', 'countries ', 'legal ', 'poet ', 'buildings ', 
		'employment', 'west ', 'little ', 'al ', 'against', 'argument ', 'knowledge ', 'by ', 'sense ', 'political ', 'models ',
		'belief ', 'human ', 'historical ', 'greater ', 'state ', 'universe ', 'philosophy ', 'humans ', 'beings ', 'evidence ',
		'scientists ', 'thank ', 'leap', 'complicated ', 'literature ', 'particularly ', 'prayers ', 'giveaway ', 'thankful ', 
		'hubby ', 'let ', 'unlikely ', 'less ', 'complex ', 'folk ', 'terms ', 'fucking ', 'entirely ', 'structure ', 'cultural ', 
		'liberal ', 'university ', 'bizarre ', 'imaginative', 'fancy', 'fantasy', 'imaginary', 'visionary', 'fanciful', 'imagine',
		'fantastic', 'creative', 'earthbound', 'fiction', 'relive', "mind", 'ideal', 'prosaic', 'fire', 'conceit', 'imagery',
		'phantasm', 'invention', 'vision', 'artistic', 'imaginational', 'pedestrian', 'misimagination', 'art', 'imaginal', 
		'dreamland', 'chimera', 'enterprising', 'creation', 'conceited', 'purblind', 'nonentity', 'phantasy', 'sterile', 'fertile',
		'fictive', 'dream', 'hack', 'fictitious', 'create', 'poetry', 'makebelieve', 'brain', 'otherworldly', 'step', 'spin', 
		'invent', 'romantic', 'associate', 'artist', 'fine art', 'stodgy', 'dryness', 'unimaginative', 'dreamworld', 'project',
		'dream', 'association', 'fancier', 'fancied', 'exalt', 'imaginability', 'ingenuity', "in one's mind's eye", 'figment',
		'suspicion', 'romanticism', 'imaginatively', 'picture', 'imaginings', 'poet', 'arid', 'ideally', 'fume', 'devise', 'myth',
		'madeâ€“up', 'vivid', 'reproduce', 'image', 'mythical', 'conceive', 'inventive', 'foreshorten', 'soar', 'evoke', 'prosy',
		'vapid', 'cloudland', 'daydream', 'fertility', 'seize', 'verve', 'flight', 'presentive', 'fanciless', 'surmise', 'scene',
		'concoct', 'thinking', 'reproductive', 'imagination', 'catch', 'creativity', 'suffocate', 'referent',
		'dreamworld', 'visualize', 'poem', 'flighty', 'impress', 'mind', 'strong', 'shortsighted', 'wit', 'boilerplate',
		'capture', 'whimsical', 'connection', 'riot', 'species', 'prose', 'apprehension', 'slave', 
		'flame', 'obvious']

	#Conscientiousness
	consc= ['fired', 'Roberts ', 'rough ', 'Hawaii', 'desperate ', 'routine ', 'tbsp ', 'vegetables ', 'garlic ', 'temperature ', 'carrots', 
		'melted', 'salad ', ' popcorn ', ' days', 'terror', 'jail ', 'warm', 'enjoying ', 'with ', 'extreme ', 'cheese ', 'rest ',
		'intelligent ', 'deck ', 'bang ', 'pity', 'lots ', 'stack ', ' finished ', 'pathetic ', 'visit ', 'stupid ', 'idiot ',
		'religious ', 'vain ', 'decent', 'wallet ', 'deny', 'rarely ', 'bloody', 'protest', 'utter ', 'contrary', 'shame', 'majority', 'soldiers',
		'drunk', 'politically', 'democracy', 'entirely ', 'practical ', 'ready', 'HR', 'rarely', 'boring ', 'quality', 'overcome ', " mom ",
		'until', 'clever', 'Mexican ', ' pace ', 'challenging ', 'addition', 'anxious', 'jokes', 'paid ', 'self', 'pride', 'ego', 'proud',
		'egoism', 'inferiority', 'complex', 'proper', 'worth', 'respect', 'flown', 'indignity',
		'humiliation', 'puff', 'greatness', 'idiolatry', 'respecting', 'dignified', 'abasement', 'place', 'respect', 'disesteem', 'regard',
		'estimation', 'uppity', 'puncture', 'dysthymia', 'massage', 'deflate', 'egotism', 'autotheism', 'pompous', 'value', 'conceit', 'narcissism',
		'esteem', 'pride', 'estimation', 'pique', 'shake', 'selves', 'supportive' ,'therapy', 'peg', 'proudly', 'humble',
		'significant', 'take', 'modesty', 'down', 'promotion', 'bloated', 'prize', 'honor', 'negatively', 'nervous', 'breakdown',
		'firmness', 'consideration', 'estimable', 'misesteem', 'upstart', 'regard', 'reputation', 'abase', 'consequential', 'sell', 'hold',
		'honour', 'credit', 'codependency', 'cripple', 'respectable', 'undervalue', 'cheap', 'bumptious', 'autogenous', 'autonomy', 'disant',
		'autogenetic', 'destroy', 'depressive', 'disorder', 'government', 'adoration', 'onanism', 'autolatry', 'overrate', 'possessed',
		'count', 'autogeneal', 'disparage', 'brass', 'consider', 'creditable', 'auto', 'let', 'contained', 'righteous', 'cocky',
		'existimate', 'launderette', 'command', 'sufficient', 'deceit', 'win', 'opinion', 'acceptation',
		'honorific', 'herself', 'collected', 'induced', 'valuable', 'selfism', 'debase', 'ascesis',
		'admire', 'compliment', 'disrepute', 'aplomb', 'meritorious', 'testimonial', 'beloved', 'dignity', 'confidence', 'denial',
		'abuse', 'autarchy', 'autodidact', 'premium', 'respected', 'pedestal', 'estimate', 'autogenic', 'abnegation',
		'reputable', 'yourself', 'assured', 'address', 'love', 'think', 'sufficiency', 'bless', 'adore', 'autogamy', 'myself',
		'himself', 'mastery', 'existimation', 'cockalorum', 'reckoning', 'taught', 'egotist', 'truism', 'homage', 'egotistic',
		'demean', 'solid', 'assured', 'autokinesy', 'control', 'automotive', 'confident', 'worship', 'fellow', 'have', 'judge',
		'restraint', "feet", 'acquit', 'autofluorescence', 'automorphic', 'big', 'flatulent', 'reverence', 'considered',
		'good']


	#Extraversion
	extra= ['bar', 'other', 'drinks', 'restaurant', 'dancing ', 'restaurants ', 'cats', 'grandfather  ', 'Miami  ', 'countless  ', 'drinking  ', 'shots  ', 
		'computer  ', 'girls  ', 'glorious ', 'minor ', 'pool  ', 'crowd ', 'sang  ', 'grilled', 'house', 'clothes', 'money', 'dining', 'travel', 'car', 'drink', 
		'table', 'clothes', 'bottle', 'trip', 'lunch', 'father', 'mother', 'brother', 'sister', 'singing', 'swimming', 'laptop', 'talkative', 'social', 'congenial',
		'gregarious', 'personable', 'sociable', 'cordial', 'demonstrative', 'friendly', 'adaptability', 'companionship', 'boldness ', 'brashness', ' forwardness ', 
		'immodesty', 'camaraderie', 'companionship', 'fellowship', 'amiability', 'cordiality ', 'folksiness', 'friendliness', 'neighborliness', 'conviviality', 
		'gregariousnessy', 'friends ', 'leisure ', ' family ', ' up ', 'social', ' positive', 'emotions  ', 'sexual  ',
		'space ', 'physical', ' home ', 'sports ', 'motion  ', 'music ', 'inclusive ', 'eating ', 'time ', 'optimism ', 'causation', 'music ',
		'feelings ', 'affect  ', 'friends ', 'sexual', 'Leisure  ', 'states ', 'assent ', 'perfect ', ' denied ', ' sweet', 'song  ', 'every  ',
		'temporary ', ' dance  ', 'golden ', 'Openness', 'sang', 'hotel ', 'lazy ', 'kissed', 'shots ', 'golden ', 'dad', 'girls ', 'restaurant ', 'eve',
		'best ', 'proud', 'accept ', 'soccer', 'met', 'not', 'brothers', 'interest', 'cheers','bonhomie', 'amicable', 'warmth', 'affable', 'fellowship', 
		'thaw', 'sociality', 'sociable', 'friending', 'greeting', 'friendling', 'friendliness',
		'genial', 'cordial', 'cool', 'lovefeast', 'coolness', 'geniality', 'wink', 'friendlihood', 'goodwill', 'amiable', 'friendly', 'repulsive', 'chum',
		'estrange', 'empressement', 'unapproachable', 'hearty', 'peace', 'stony', 'amicability', 'familiarity', 'welcome', 'will',
		'community', 'spirit', 'biofriendly', 'distance', 'companiable', 'treat', 'unfriendly', 'glacial', 'amity', 'cold', 'mistake', 'ice', 'intimate', 'veneer',
		'kindliness', 'reconcile', 'smoothfaced', 'coolly', 'conviviality', 'warm', 'regard', 'comity', 'alien', 'handshaker', 'alienate', 'sympathetic', 
		'affability', 'marblehearted', 'camaraderie', 'jovial', 'brittle', 'disaffect', 'unchanging', 'greenwash',  'noticeable', 'unctuous',
		'personality', 'normalize', 'fawn', 'smile', 'courteous', 'abatement', 'lovable', 'clubby', 'subderivative', 'cordially', 'hospitality', 'congeniality',
		'synthetic', 'loving', 'gamble', 'gene', 'macho', 'machismo', 'spine', 'forwardness', 'ipsedixitism', 'crust', 'bluster', 'spirit', 'unaggressive',
		'belligerent', 'positivity', 'positivism', 'tough', 'shyness', 'adventurous', 'thrill', 'delirium', 'exciting', 'fever', 'chase','thrill',
		'electricity', 'sensation', 'orgasm', 'flutter', 'seeking', 'intoxication', 'stir', 'heat', 'calm', 'fermentation', 'frenzy', 'twitter', 
		'inquiry', 'rapture', 'buzz', 'breathless', 'purchase', 'flurry']

	op= calcy(tokens,openn)
	openness= (op/lengy)*10
	openness=round(openness,2)

	c= calcy(tokens,consc)
	consciousness= (c/lengy)*10
	consciousness=round(consciousness,2)

	e= calcy(tokens,extra)
	extraversion= (e/lengy)*10
	extraversion=round(extraversion,2)

	a= calcy(tokens,agree)
	agreeableness =(a/lengy)*10
	agreeableness=round(agreeableness,2)

	n= calcy(tokens,neuro)
	neuroticism= (n/lengy)*10
	neuroticism=round(neuroticism,2)


	per_name="static/report/"+my_name+"/Personality-"+my_name+".png"
	data = {'Openness':openness, 'Conscientious':consciousness, 'Extraversion':extraversion,'Agreeableness':agreeableness, 'Neuroticism':neuroticism} 
	per = list(data.keys()) 
	score = list(data.values()) 
	
	fig = plt.figure() 
	
	# creating the bar plot 
	plt.bar(per, score, color ='#16c79a',width = 0.6) 

	plt.xlabel("Personality Traits") 
	plt.title("Persona Analysis") 
	plt.savefig(per_name)
	plt.close(fig)
	# plt.show()
