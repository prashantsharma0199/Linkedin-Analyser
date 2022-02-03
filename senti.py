import os, random,sys,time
from urllib.parse import urlparse
from selenium import webdriver
from bs4 import BeautifulSoup

# print("ATTENTION: Kindly enter the username of the person you want to do sentiment analysis of (in small alphabets and no spaces) ")
# time.sleep(4)

# name = input("Enter the LINKEDIN username of person: ")

# f_name= input("Enter the name you want to save your file with: ")

def findData(name, f_name, username, password):
	fily= "data/"+f_name+".txt"
	f=open(fily,"w+",encoding='utf-8')


	#selenium webdriver settings to open driver in background

	user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 Edg/87.0.664.75"

	options = webdriver.ChromeOptions()
	options.headless = True
	options.add_argument(f'user-agent={user_agent}')
	options.add_argument("--window-size=1920,1080")
	options.add_argument('--ignore-certificate-errors')
	options.add_argument('--allow-running-insecure-content')
	options.add_argument("--disable-extensions")
	options.add_argument("--proxy-server='direct://'")
	options.add_argument("--proxy-bypass-list=*")
	options.add_argument("--start-maximized")
	options.add_argument('--disable-gpu')
	options.add_argument('--disable-dev-shm-usage')
	options.add_argument('--no-sandbox')



	#setting up a webdriver 
	browser = webdriver.Chrome(executable_path='chromedriver.exe', options=options)

	browser.get('https://www.linkedin.com/uas/login')


	#reading login credentials from file to login into linkedin
	# file = open('credentials.txt')
	# lines=file.readlines()
	# username=lines[0]
	# password=lines[1]

	#filling the login credentials on linkedin
	elementID = browser.find_element_by_id('username')
	elementID.send_keys(username)

	elementID = browser.find_element_by_id('password')
	elementID.send_keys(password)

	time.sleep(3)
	elementID.submit()

	# name="radupalamariu"
	link="https://www.linkedin.com/in/"+name+"/detail/recent-activity/shares/"

	browser.get(link)


	#scrolling down the entire page
	SCROLL_PAUSE_TIME=0.5

	last_height= browser.execute_script("return document.body.scrollHeight")

	for i in range(550):#500
	# while True:
		browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')

		time.sleep(SCROLL_PAUSE_TIME)

		new_height= browser.execute_script("return document.body.scrollHeight")

		if new_height== last_height:
			break

		last_height== new_height


	#capturing the element (all the caption of post)
	src = browser.page_source
	soup = BeautifulSoup(src,'lxml')


	main_div= soup.find('div',{'class':'voyager-feed'})
	feed= soup.find_all('div',{'class':'feed-shared-text'})
	# print(feed)

	browser.quit()

	#writing the data captured above in file
	for i in range(1000):
		try:
			caption= feed[i].find('span').get_text().strip()
			print(caption)
			f.write(caption+"\n")
			print('\n')
		except IndexError as e:
			break
		

	f.close()
