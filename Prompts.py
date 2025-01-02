role = '''I have clustered some data relating to the persona of different founders into clusters and sub-clusters of each cluster. You, an LLM, are being used as a post-hoc explainer to give a written explanation for each cluster and sub-cluster so that I can understand each cluster and the persona of the founder that belongs to it.''' 

meta_prompt = ''' I have clustered some data relating to the persona of different founders into clusters and sub-clusters of each cluster. I am using chatgpt API as a post-hoc explainer to give a written explanation for each cluster and sub-cluster so that I can understand each cluster and the persona of the founder that belongs to it. I want the output to be in three columns: Firstly it should an couple/few word overview of each cluster (e.g. high entrepreneur experience), then it should give a more detailed explanation of the persona in the cluster (e.g. well educated at a good institution and a risk taker), then finally it should give precise data backed evidence for the cluster (e.g. went to a top 10 institution and has the highest level of IPO experience). '''

data_explanation = '''this is the explanation of each category, use this to update your brief description and make it more precise: **Educational Background**: 
- education_level: Assign scores based on the highest degree attained - 3 (Doctorate or Professional Degree), 2 (Master's Degree), 1 (Bachelor's Degree), 0 (Associate Degree or No Degree) 
- education_institution: Give this a thought and try to rank the best educational institution of the founder based on the founder’s profile using QS World University Ranking 2023 - 4 (University ranked in the top 20), 3 (University ranked 20-100), 2 (University ranked 100-500), 1 (University ranked 500-2000), 0 (Unknown or University ranked after 2000) 
- education_field_of_study: Check the subject of their most advanced degree on their profile and determine what field they studied - 3 (STEM), 2 (Business, Economics, Finance), 1 (Social Sciences), 0 (Other)
education_international_experience:
Evaluate if the founder has studied abroad or at an international institution outside of the US:
1 (Yes)
0 (No)
education_publications_and_research:
Check if the founder has any publications or research experience:
1 (Yes, has publications or significant research experience)
0 (No)
education_extracurricular_involvement:
Assess involvement in extracurricular activities, especially leadership roles (e.g., clubs, organizations):
1 (High involvement)
0 (Low or no involvement)
education_awards_and_honors:
Identify if the founder received any notable awards or honors during their education such as Valedictorian, first-class honors, 4.0 GPA:
1 (Yes, has received awards or honors)
0 (No)
**Industry Experience**: 
Do NOT consider mere internships, part-time roles, or teaching/research assistant roles. ONLY consider full-time roles in the industry!
- years_of_experience: The number of years since the first full-time role in the industry. Do not consider internships or part-time or teaching/research assistant roles or other mere roles in the industry. You can typically calculate the number of years of experience from the time that they graduated to $founding_year. You can calculate the difference. 0 (0-5 years experience), 1 (5-10 years experience), 2 (10-15 years experience), 3 (15-20 years experience), 4 (20-30 years experience), 5 (more than 30 years experience).
- number_of_roles: The total number of full-time roles held in the industry. If a founder moved up and changed roles in the same company, you can count them as well. 0 (no roles), 1 (1 role), 2 (2 roles), 3 (3 roles), 4 (4 roles), 5 (5 roles), 6 (6 or more than 6 roles)
- number_of_companies: The total number of companies that they worked full-time for. Be careful if they have listed multiple positions at the same company. These should only count as one company. Also be careful about people listing internships, teaching-assistant, research assistant, angel investor, jobs at university and other part time jobs as experience these shouldn't count as additional companies. Make sure the number of companies only includes full-time roles. 0 (no companies before), 1 (1 company), 2 (2 companies), 3 (3 companies), 4 (4 companies), 5 (5 companies), 6 (6 or more than 6 companies)
- industry_achievements: Number of significant/notable achievements or awards in the industry. 0 (no achievements), 1 (1 achievement), 2 (2 achievements), 3 (3 achievements), 4 (4 or more than 4 achievements)
- big_company_experience: 1 (Experience at a Fortune 500 company), 0 (No experience at a Fortune 500 company)
- nasdaq_company_experience: 1 (if they worked in a public tech company), 0 (no experience in public tech company). Public tech company can be defined as any company that is in the NASDAQ 100 Technology Sector (NDXT) when they work at that company. 
- big_tech_experience: 1 (if they worked in Alphabet/Google, Amazon, Apple, Meta, Microsoft), 0 (no experience in Alphabet/Google, Amazon, Apple, Meta, Microsoft).
- google_experience: 1 (if they worked at Google), 0 (did not work at Google). 
- facebook_meta_experience: 1 (if they worked at Facebook/Meta), 0 (did not work at Facebook/Meta)
- microsoft_experience: 1 (if they worked at Microsoft), 0 (did not work at Microsoft)
- amazon_experience: 1 (if they worked at Amazon), 0 (did not work at Amazon)
- apple_experience: 1 (if they worked at Apple), 0 (did not work at Apple)
- big_tech_position: check if they worked as an engineer, researcher, product manager or other roles at a Big Tech company (Alphabet/Google, Amazon, Apple, Meta, Microsoft). BigTech companies are as defined above. Make sure this position is at BigTech company. If they have worked as one of these positions in a non BigTech company it doesn’t count and that should be marked as 0. Please think through what this profile belongs to among researcher/engineer/product manager/other/non-BigTech, and provide the category. Researcher: 5, Engineer: 4, Product manager: 3, Sales & Marketing: 2, Other: 1, non-BigTech: 0
- number_of_companies: Count of unique companies they've worked for in a full-time capacity
- career_growth: 2 (Significant growth in founder profile, e.g. software engineer to VP of engineering in a top tech company), 1 (growth in the founder profile, but not as strong such as engineering to engineering manager), 0 (No strong career growth is observed)
- moving_around: 1 (founder changes full-time jobs too frequently. Example is that he works 1 year in company X. Then he works at another company Y for 1 year.). 0 (founder has no instance of too much of a frequent job change. He appears to be stable.)
international_work_experience:
1 (If they have worked full-time in multiple countries by looking into the country of the jobs)
0 (No international work experience)
- worked_at_military: 1 (worked at military) and 0 (no military experience)
- worked_at_consultancy: 3 (worked at McKinsey, Boston Consultant Group or Bain), 2 (worked at mid-tier consultancies such as Accenture), 1 (worked at unknown consultancies), 0 (never worked in a consultancy)
- worked_at_bank: 3 (worked at a top-tier bank such as Goldman Sachs), 2 (worked at mid-tier large bank such as Bank of America), 1 (worked at unknown or medium size banks), 0 (did not work at a bank)
**Technical Skills and Patents**:
- patents_inventions: 0 (No patents), 1 (1-10 patents), 2 (More than 10 patents)
- technical_skills: technical_skills: 0 (No engineering or technical skills), 1 (Engineering or technical skills evident from education or job roles with technical responsibilities)
Technical_publications: Count the number of technical publications (papers, articles) authored by the founder. 0 (No publications), 1 (1-10 publications), 2 (More than 10 publications)
Technical_leadership_roles: Identify if the founder has held technical leadership roles such as CTO, Head of Engineering or distinguished engineer. 1 (Yes), 0 (No)
**Leadership Roles**:
- big_leadership: 3 (held c-level roles at Fortune 500 companies), 2 (held VP roles at Fortune 500 companies), 1 (Held Director at Fortune 500 companies), 0 (No leadership roles at Fortune 500 companies).
- nasdaq_leadership: 3 (held c-level roles at public tech companies), 2 (held VP roles at public tech companies), 1 (held director roles at public tech companies), 0 (no leadership roles at public tech companies). Public tech company can be defined as any company that is in the NASDAQ 100 Technology Sector (NDXT) when they worked at that company. 
- bigtech_leadership: 3 (held c-level roles at BigTech), 2 (Held VP roles at BigTech), 1 (Held director roles at BigTech ), 0 (no leadership roles at BigTech). BigTech is defined as Alphabet/Google, Amazon, Apple, Meta, Microsoft.
- number_of_leadership_roles: 0 (No leadership roles), 1 (Held one leadership role before starting the business), 2 (held more than one leadership role before starting the business).
- being_lead_of_nonprofits: 1 (Being a leader in non-profits), 0 (No leadership roles)
**Startup Experience**:
- startup_experience: 0 (No startup experience), 1 (worked at a startup)
- previous_startup_funding_experience_as_ceo: The founder raised money for his own startup as its founder and CEO. 0 (not available), 1 (funding less than 3M), 2 (3M-10M in funding), 3 (10M-50M in funding), 4 (More than 50M in funding)
- previous_startup_funding_experience_as_nonceo: The founder raised money for his own startup as one of its founders (but not as its CEO). 0 (not available), 1 (funding less than 3M), 2 (3M-10M in funding), 3 (10M-50M in funding), 4 (More than 50M in funding)
- ceo_experience: 0 (No CEO experience), 1 (Has been a CEO before the founder started this new company). Please think through and make sure that the founder had a prior CEO experience before starting this new company. The title you see should be CEO or Chief Executive Officer. If that doesn’t exist, you should mark this as 0.  
- founder_experience: 0 (Has not founded a company), 1 (Has founded or co-founded a company), 2 (has founded or co-founded more than one company)
- prior_ipo_experience: 0 (no prior startup exists, so not available), 1 (founder took a previous company to public via IPO as its founder)
- prior_acquisition_experience: 0 (no prior startup exists, so not available), 1 (the founder’s prior company was not acquired), 2 (the founder’s prior company was acquired, no amount is disclosed), 3 (founder’s prior company was acquired less than $20M), 4 (founder’s prior company was acquired more than $20M and less than $50), 5 (founder’s prior company was acquired more than $50M and less than $150M), 6 (founder’s prior company was acquired more than $150M and less than $500M), 7 (founder’s prior company was acquired more than $500M)
- prior_acquirer_bigtech: Mark 1 if the founder’s prior company was acquired by Alphabet/Google, Amazon, Apple, Meta, Microsoft, otherwise mark as 0. 
- investor_quality_prior_startup: 0 (no prior startup exists, so not available), 1 (prior startup exists, it raised money, but not from tier-1 VCs), 2 (prior startup raised money from tier-1 VCs). Tier-1 VCs are 'Greylock', 'Benchmark', 'Foundation Capital', 'Floodgate', 'Lowercase Capital', 'Accel', 'Sequoia Capital', 'Redpoint', 'Kleiner Perkins', 'GV', 'Lightspeed Venture Partners', 'First Round Capital', 'General Catalyst', 'True Ventures', 'Founders Fund', 'Andreessen Horowitz', 'Union Square Ventures', 'Felicis Ventures', 'Battery Ventures', 'CRV', 'Menlo Ventures', 'Mayfield Fund', 'DCVC', 'New Enterprise Associates', 'Lux Capital', 'Khosla Ventures', 'Intel Capital'.
**Investment Experience**
- VC_experience: 0 (founder did not hold a role at a VC firm), 1 (founder had a junior role in a VC firm), 2 (founder had a senior role in a VC firm)
- tier_1_VC_experience: 0 (founder did not work at a tier-1 VC firm), 1 (founder worked at a tier-1 VC firm). Tier-1 VC firms are 'Greylock', 'Benchmark', 'Foundation Capital', 'Floodgate', 'Lowercase Capital', 'Accel', 'Sequoia Capital', 'Redpoint', 'Kleiner Perkins', 'GV', 'Lightspeed Venture Partners', 'First Round Capital', 'General Catalyst', 'True Ventures', 'Founders Fund', 'Andreessen Horowitz', 'Union Square Ventures', 'Felicis Ventures', 'Battery Ventures', 'CRV', 'Menlo Ventures', 'Mayfield Fund', 'DCVC', 'New Enterprise Associates', 'Lux Capital', 'Khosla Ventures', 'Intel Capital'.
- angel_experience: 0 (founder made no angel investments), 1 (founder made between 1 to 10 angel investments), 2 (founder made more than 10 investments)
- quant_experience: 0 (founder was not a quant investor), 1 (founder was a quant at an investment firm), 2 (founder was a quant at a reputable investment firm like Bridgewater or Renaissance Technologies). 

- board_advisor_roles: 0 (No board or advisor roles or only at small/unknown companies), 1 (Held board or advisor roles at large companies or well-known startups)

**Influence**:
- press_media_coverage_count: A numerical indicator of the level of press or media coverage received by an individual, excluding any coverage related to their companies. 0 (No significant press or media coverage), 1 (Moderate press or media coverage. he individual has received limited press attention, either through a single feature in a high-profile media outlet or multiple features in less prominent outlets), 2 (The individual has been featured multiple times in high-profile media outlets, establishing a notable public presence).
- significant_press_media_coverage: 0 (No press or media coverage), 1 (More than 1 press and media coverage)
- speaker_influence: 0 (no events attended as a speaker).  1 (attended between 1 and 3 events as a speaker). 2 (attended more than 3 events as a speaker). 
**Childhood Mastery**
professional_athlete: 0 (founder was not a professional athlete), 1 (founder was a professional athlete in the high school or university team).
languages: 0 (founder did not report any languages that he speaks), 1 (founder can speak 2 languages), 2 (founder can speak 3 languages), 3 (founder can speak more than 4 languages)
childhood_entrepreneurship: 0 (no proof of entrepreneurship during university years or before university times), 1 (evident proof of entrepreneurship in founder’s Linkedin profile at a young age before the age 22)
ten_thousand_hours_of_mastery: 0 (the founder doesn’t have evident proof in his profile that he never spent 10K hours to master an area), 1 (founder has reasonable proof of time commitment for an area that he mastered, and this area is technically really hard and competitive. This could be a Phd or being a grandmaster in chess or attending olympiads as a participant.)
Competitions: 0 (founder did not or partook in less than 2 competitions), 1 (founder participated in more than 2 competitions)
**Characteristics**
Use best of your ability to conclude what’s written on the profile to assess the following soft skills and characteristics:
Extroversion: If the founder wrote detailed descriptions in the jobs and has indication of high usage of Linkedin, then it indicates extroversion. If the founder has no description and incomplete profile, then that is low extroversion. 
2 (High extroversion)
1 (Low extroversion)
0 (No evidence of extroversion, inconclusive)
Perseverance: The founder should have a clear track record of sticking with projects and roles through challenges, with descriptions highlighting overcoming significant obstacles (e.g., "Led the company through a tough financial period, ultimately securing a successful Series B round"). This is high perseverance. 
When the founder has frequently changed jobs, with many roles lasting less than a year, and descriptions lack details on challenges or long-term achievements, this is low perseverance. 
2 (Demonstrated high perseverance)
1 (demonstrated low perseverance)
0 (No evidence of perseverance, inconclusive)
Risk Tolerance: Willingness to take risks in their career or business ventures. The founder left a senior position at a Fortune 500 company to start a fintech startup targeting an unproven market segment is a good example of high risk tolerance. If the founder has spent their career at large, stable companies, progressing through well-defined roles without any involvement in startups or high-risk projects, that is low risk tolerance.
2 (High risk tolerance)
1 (Low risk tolerance)
0 (No evidence of risk tolerance, inconclusive)
Vision: Highlighting the ability to see the bigger picture and work or lead on innovative projects and technologies before starting a company is a signal of vision. If the founder managed several projects in established industries, primarily focusing on maintaining and optimizing existing systems without driving major innovations, that is a negative signal for vision. These profiles are dependable and detail-oriented but do not mention any visionary qualities or forward-thinking achievements.
1 (Vision-oriented experiences)
2 (No vision-oriented experiences)
0 (No clear evidence, inconclusive)
Adaptability: Highlighting the ability to quickly adapt to new roles and environments, emphasizing the flexibility and willingness to embrace change is a positive signal. For instance, transitioning from a technical role in a traditional industry to a leadership position in a fast-paced tech startup is a signal of adaptability. If the founder has spent their career in a single industry, holding similar roles across different companies with little variation in responsibilities or challenges, that is negative for adaptability. 
2 (High adaptability)
1 (Low adaptability)
0 (no clear evidence, inconclusive) 
Emotional intelligence: Look for the following indicators in the descriptions:
Empathy and Understanding: Mentions of understanding others' needs, showing empathy, or supporting team members (e.g., "led with empathy").
Mentorship and Coaching: References to mentoring or coaching others (e.g., "mentored junior staff").
Conflict Resolution and Team Dynamics: Examples of managing conflicts, improving collaboration, or fostering a positive environment (e.g., "resolved conflicts").
Leadership in Challenging Situations: Descriptions of leading through tough times while maintaining morale and cohesion (e.g., "navigated market downturns").
Diversity and Inclusion: Involvement in diversity and inclusion efforts (e.g., "led diversity initiatives").
Collaborative Language: Use of language that emphasizes teamwork and collective success (e.g., "collaborated with the team").
Handling Difficult Situations: Managing difficult conversations or providing feedback sensitively (e.g., "handled sensitive feedback").
Assign a score: 2 for high emotional intelligence (clear evidence of these traits), and 1 for low emotional intelligence (low evidence of of these traits), 0 (no clear evidence, inconclusive). 
Personal Branding: Evaluate the personal branding of a founder based on the job and education descriptions in their LinkedIn profile. Focus on identifying the following indicators: 
Public Speaking or Thought Leadership: Does the founder mention speaking engagements, industry panels, or thought leadership activities? 
Published Work or Media Appearances: Are there mentions of published articles, books, or media contributions? 
Leadership Roles with Public Focus: Does the founder hold or mention roles involving external engagement, such as being a spokesperson or leading PR efforts? 
Recognition and Awards: Are there awards or recognitions specifically related to public visibility or influence? 
Initiatives that Increase Visibility: Does the founder mention initiating or leading projects that enhance their visibility or public profile? 
Assign a score: 1 for strong personal branding (clear evidence of these efforts), and 0 for weak or minimal personal branding (little to no evidence).'''


print('successful Prompt import')