import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv('tmdb_5000_movies.csv')
df2 = pd.read_csv('tmdb_5000_credits.csv')

s = set()
for j in df['genres']:
    d = json.loads(j)
    for i in d:
        s.add(i['name'])
genres = list(s)

df = df.drop(columns = ['vote_average', 'vote_count', 'tagline', 'status', 'popularity', 'original_title', 'homepage', 'production_countries', 'spoken_languages'])

day = []
month = []
year = []

for i in df['release_date']:
    try:
        day.append(int(i[:2]))
        month.append(int(i[3:5]))
        year.append(int(i[-4:]))
    except:
        day.append(0)
        month.append(0)
        year.append(0)

df['month'] = np.array(month)
df['year'] = np.array(year)
df = df.dropna()

df = df[df.year != 0]
df = df[df.month != 0]

genres_budget = {}
genres_revenue = {}
genres_ratio = {}
genres_count = {}

for i in genres:
    df[i] = ''
    genres_budget[i] = 0
    genres_revenue[i] = 0
    genres_ratio[i] = 0
    genres_count[i] = 0
    
for movie in range(df.shape[0]):
    try:
        bud = df['budget'][movie]
        rev = df['revenue'][movie]
        l = df['genres'][movie]
        movie_genres = json.loads(l)
        for i in genres:
            df.at[movie, i] = 0
        for i in movie_genres:
            name = i['name']
            genres_budget[name] += bud
            genres_revenue[name] += rev
            genres_count[name] += 1
            df.at[movie, name] = 1
    except:
        print(end = '')

for i in genres_budget:
    if(genres_budget[i] != 0 and genres_revenue[i] != 0):
        x = genres_revenue[i]/genres_budget[i]
        if(not pd.isna(x)):
            genres_ratio[i] = genres_revenue[i]/genres_budget[i]
        
pc_budget = {}
pc_revenue = {}
pc_ratio = {}
pc_count = {}

for movie in range(df.shape[0]):
    try:
        bud = df['budget'][movie]
        rev = df['revenue'][movie]
        l = df['production_companies'][movie]
        d = json.loads(l)
        if len(d) > 0:
            pc = d[0]['name']
            if pc not in pc_budget:
                pc_budget[pc] = bud
                pc_revenue[pc] = rev
                pc_count[pc] = 1
            else:
                pc_budget[pc] += bud
                pc_revenue[pc] += rev
                pc_count[pc] += 1
    except:
        print(end = '')

for i in pc_budget:
    if(pc_budget[i] != 0 and pc_revenue[i] != 0):
        x = pc_revenue[i]/pc_budget[i]
        if(not pd.isna(x)):
            pc_ratio[i] = pc_revenue[i]/pc_budget[i]
            
cast = []
for i in range(df.shape[0]):
    try:
        x = df2.loc[df2['movie_id'] == df['id'][i]]
        l = list(x['cast'])
        d = json.loads(l[0])
        movie_cast = []
        for j in range(min(5, len(d))):
            movie_cast.append(d[j]['name']) 
        cast.append(movie_cast)
    except:
        cast.append([])
cast = np.array(cast, dtype = object)

df['Cast'] = cast

actors_budget = {}
actors_revenue = {}
actors_ratio = {}
actors_count = {}

for movie in range(df.shape[0]):
    try:
        bud = df['budget'][movie]
        rev = df['revenue'][movie]
        l = df['Cast'][movie]
        for i in l:
            if i not in actors_budget:
                actors_budget[i] = bud
                actors_revenue[i] = rev
                actors_count[i] = 1
            else:
                actors_budget[i] += bud
                actors_revenue[i] += rev
                actors_count[i] += 1
    except:
        print(end = '')
        
for movie in range(df.shape[0]):
    try:
        bud = df['budget'][movie]
        rev = df['revenue'][movie]
        l = df['Cast'][movie]
        for i in l:
            if i not in actors_budget:
                actors_budget[i] = bud
                actors_revenue[i] = rev
            else:
                actors_budget[i] += bud
                actors_revenue[i] += rev
    except:
        print(end = '')
        
for i in actors_budget:
    if(actors_budget[i] != 0 and actors_revenue[i] != 0):
        x = actors_revenue[i]/actors_budget[i]
        if(not pd.isna(x)):
            actors_ratio[i] = actors_revenue[i]/actors_budget[i]

cast_attrs = ['Cast1_Score', 'Cast2_Score', 'Cast3_Score', 'Cast4_Score', 'Cast5_Score']
for i in cast_attrs:
    df[i] = ''
    
for movie in range(df.shape[0]):
    try:
        l = df['Cast'][movie]
        for i in range(5):
            df.at[movie, cast_attrs[i]] = 0
        for i in range(len(l)):
            df.at[movie, cast_attrs[i]] = actors_ratio[l[i]]
    except:
        print(end = '')
        
keywords_budget = {}
keywords_revenue = {}
keywords_ratio = {}
keywords_count = {}

for movie in range(df.shape[0]):
    try:
        bud = df['budget'][movie]
        rev = df['revenue'][movie]
        l = df['keywords'][movie]
        d = json.loads(l)
        for i in d:
            keyword = i['name']
            if keyword not in keywords_budget:
                keywords_budget[keyword] = bud
                keywords_revenue[keyword] = rev
                keywords_count[keyword] = 1
            else:
                keywords_budget[keyword] += bud
                keywords_revenue[keyword] += rev
                keywords_count[keyword] += 1
    except:
        print(end = '')
        
for i in keywords_budget:
    if(keywords_budget[i] != 0 and keywords_revenue[i] != 0):
        x = keywords_revenue[i]/keywords_budget[i]
        if(not pd.isna(x)):
            keywords_ratio[i] = keywords_revenue[i]/keywords_budget[i]
            
df['Keywords_Score'] = ''

for i in range(df.shape[0]):
    try:
        l = df['keywords'][i]
        d = json.loads(l)
        scores = []
        for j in d:
            keyword = j['name']
            #print(keyword)
            if(keyword in keywords_ratio):
                scores.append(keywords_ratio[keyword])
        scores.sort(reverse = True)
        lim = min(10, len(scores))
        if(lim == 10):
            score = sum(scores[:lim])
        else:
            score = (sum(scores[:lim]) / lim) * 10
        df.at[i, 'Keywords_Score'] = score
    except:
        print(end = '')
        
df['PC_Score'] = ''

for movie in range(df.shape[0]):
    try:
        l = df['production_companies'][movie]
        d = json.loads(l)
        if len(d) > 0:
            pc = d[0]['name']
            if pc in pc_ratio:
                df.at[movie, 'PC_Score'] = pc_ratio[pc]
            else:
                df.at[movie, 'PC_Score'] = 0
    except:
        print(end = '')
        
df['Director'] = ''
df['Producer'] = ''

for i in range(df.shape[0]):
    try:
        x = df2.loc[df2['movie_id'] == df['id'][i]]
        l = list(x['crew'])
        d = json.loads(l[0])
        for j in d:
            if(j['job'] ==  'Director'):
                df.at[i, 'Director'] = j['name']
        for j in d:
            if(j['job'] ==  'Producer'):
                df.at[i, 'Producer'] = j['name']
    except:
        print(end = '')
        
producers_ratio = {}
producers_budget = {}
producers_revenue = {}
producers_count = {}
directors_ratio = {}
directors_budget = {}
directors_revenue = {}
directors_count = {}

for movie in range(df.shape[0]):
    try:
        bud = df['budget'][movie]
        rev = df['revenue'][movie]
        dire = df['Director'][movie]
        pro = df['Producer'][movie]
        if not pd.isna(dire):
            if dire not in directors_budget:
                directors_budget[dire] = bud
                directors_revenue[dire] = rev
                directors_count[dire] = 1
            else:
                directors_budget[dire] += bud
                directors_revenue[dire] += rev
                directors_count[dire] += 1
        if not pd.isna(pro):
            if pro not in producers_budget:
                producers_budget[pro] = bud
                producers_revenue[pro] = rev
                producers_count[pro] = 1
            else:
                producers_budget[pro] += bud
                producers_revenue[pro] += rev
                producers_count[pro] += 1
    except:
        print(end = '')
        
for i in producers_budget:
    if(producers_budget[i] != 0 and producers_revenue[i] != 0):
        x = producers_revenue[i]/producers_budget[i]
        if(not pd.isna(x)):
            producers_ratio[i] = producers_revenue[i]/producers_budget[i]
            
for i in directors_budget:
    if(directors_budget[i] != 0 and directors_revenue[i] != 0):
        x = directors_revenue[i]/directors_budget[i]
        if(not pd.isna(x)):
            directors_ratio[i] = directors_revenue[i]/directors_budget[i]
            
df['Producer_Score'] = ''
df['Director_Score'] = ''

for movie in range(df.shape[0]):
    try:
        dire = df['Director'][movie]
        pro = df['Producer'][movie]
        if not pd.isna(dire):
            if dire in directors_ratio:
                df.at[movie, 'Director_Score'] = directors_ratio[dire]
            else:
                df.at[movie, 'Director_Score'] = 0
        if not pd.isna(pro):
            if pro in producers_ratio:
                df.at[movie, 'Producer_Score'] = producers_ratio[pro]
            else:
                df.at[movie, 'Producer_Score'] = 0
    except:
        print(end = '')
        

#df1 = df[df.Cast1_Score != 0]
#df1 = df1[df1.Cast2_Score != 0]
#df1 = df1[df1.Cast3_Score != 0]
#df1 = df1[df1.Cast4_Score != 0]
#df1 = df1[df1.Cast5_Score != 0]
df1 = df[df.budget != 0]
df1 = df1[df1.runtime != 0]
df1 = df1[df1.genres != '[]']
df1 = df1[df1.revenue != 0]
df1 = df1[df1.Keywords_Score != 0]
df1 = df1[df1.PC_Score != 0]
df1 = df1[df1.Director_Score != 0]
df1 = df1[df1.Producer_Score != 0]

df1['Revenue_Class'] = ''


for i in range(df1.shape[0]):
    x = df1.iloc[i]['revenue']/df1.iloc[i]['budget']
    df1.at[i, 'Revenue_Class'] = 1
    if x > 3:
        df1.at[i, 'Revenue_Class'] = 6
    elif x > 2:
        df1.at[i, 'Revenue_Class'] = 5
    elif x > 1.1:
        df1.at[i, 'Revenue_Class'] = 4
    elif x >= 0.9:
        df1.at[i, 'Revenue_Class'] = 3
    elif x >= 0.5:
        df1.at[i, 'Revenue_Class'] = 2
        
df1 = df1.dropna()

df1.to_csv('df1.csv', index = False)

new = pd.read_csv('df1.csv')

movies = new.drop(columns = ['title', 'id', 'genres', 'keywords', 'overview', 'original_language', 'production_companies', 'release_date', 'Director', 'Producer', 'Cast'])

movies = movies.dropna()

revenue = movies['revenue']

revenue_class = movies['Revenue_Class']

movies = movies.drop(columns = ['revenue', 'Revenue_Class'])

X_train, X_test, y_train, y_test = train_test_split(movies, revenue, test_size = 0.1, random_state = 0)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

print("GBR Training Score:", reg.score(X_train, y_train))

y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score
print("GBR Testing Score:", r2_score(y_test, y_pred))

from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
X_train, X_test, y_train, y_test = train_test_split(movies, revenue_class, test_size = 0.1, random_state = 0)

clf.fit(X_train, y_train)

print("SVM Classification Training Score:", clf.score(X_train, y_train))

print("No. of Features:", movies.shape[1])

new = new.dropna()
df = new.drop(columns = ['title', 'id', 'genres', 'keywords', 'overview', 'original_language', 'production_companies', 'release_date', 'Director', 'Producer', 'Cast'])
reg.fit(df.drop(columns = ['revenue', 'Revenue_Class']), df['revenue'])
pred = reg.predict(df.drop(columns = ['revenue', 'Revenue_Class']))

upcoming = []
movie = []
movie.append([250000000, 150, 12, 2022, ['Action', 'Adventure', 'Fantasy', 'Science Fiction'], ['Sam Worthington', 'Zoe Saldana', 'Kate Winslet', 'Sigourney Weaver', 'Stephen Lang'], ['culture clash', 'space war', 'space colony', 'society', 'romance', 'tribe', 'alien planet', 'cgi', 'marine', 'love affair'], 'Twentieth Century Fox Film Corporation', 'James Cameron', 'James Cameron'])
movie.append([165000000, 147, 6, 2022, ['Adventure', 'Action', 'Science Fiction', 'Thriller'], ['Sam Neill', 'Laura Dern', 'Jeff Goldblum', 'Chris Pratt', 'Bryce Dallas Howard'], ['dna', 'monster', 'tyrannosaurus rex', 'velociraptor', 'island', 'dinosaur', 'suspense', 'disaster', 'escape', '3d'], 'Amblin Entertainment', 'Colin Trevorrow', 'Frank Marshall'])
movie.append([200000000, 105, 6, 2022, ['Animation', 'Science Fiction', 'Adventure', 'Action', 'Family'], ['Chris Evans', 'Keke Palmer', 'Peter Sohn', 'James Brolin', 'Efren Ramirez'], ['animation', 'space', 'toy story', 'space travel', 'spacecraft', 'spaceship', 'toy comes to life', 'adventure', 'robot', 'astronaut'], 'Walt Disney Pictures', 'Angus MacLane', 'Galyn Susman'])
for i in movie:
    m = []
    m.append(i[0])
    m.append(i[1])
    m.append(i[2])
    m.append(i[3])
    for j in genres:
        if j in i[4]:
            m.append(1)
        else:
            m.append(0)
    for j in i[5]:
        if j in actors_ratio:
            m.append(actors_ratio[j])
        else:
            m.append(0)
    for j in i[6]:
        x = 0
        c = 0
        if j in keywords_ratio:
            x += keywords_ratio[j]
            c += 1
    m.append((x / c) * 10)
    if i[7] in pc_ratio:
        m.append(pc_ratio[i[7]])
    else:
        m.append(0)
    if i[8] in producers_ratio:
        m.append(producers_ratio[i[8]])
    else:
        m.append(0)
    if i[9] in directors_ratio:
        m.append(directors_ratio[i[9]])
    else:
        m.append(0)
    upcoming.append(m)
upc_pred = reg.predict(upcoming)


# i = 0
# for w in sorted(pc_ratio, key=pc_ratio.get, reverse=True):
#     if(pc_count[w] > 20):
#         print(w, pc_ratio[w])
#         i += 1
#     if i == 5:
#         break
    
# i = 0
# for w in sorted(genres_ratio, key=genres_ratio.get, reverse=True):
#     print(w, genres_ratio[w])

with st.container():
    st.title('BOXOP - Box Office Prediction')
    
st.empty()
st.header('')
st.empty()
st.header('')

st.sidebar.title('BOXOP')
st.sidebar.empty()
button1 = st.sidebar.button('About the Project')
button2 = st.sidebar.button('Most Influential Factors')
button3 = st.sidebar.button('Sample Predictions')
button4 = st.sidebar.button('Upcoming Movies')
button5 = st.sidebar.button('Most Successful')
button6 = st.sidebar.button('Test It Out')

if button1:
    with st.expander('PROBLEM STATEMENT'):
        st.caption('BOXOP - The Box Office Predictor aims to predict the worldwide box office revenue of an upcoming movie based on data such as popularity, budget, release date, production company, director etc.')
        
    with st.expander('ABSTRACT'):
        st.caption('Movies have become a major form of digital entertainment since the 20th century. Most movies from major production companies generate profits above $100m even after factoring in marketing costs. The film industry is a huge commercial hub now and as in any other industry, it requires planning and strategy for maximum profit. \n\nWhile some movies may make money even in the billions, other movies may fail to turn up a profit. It is not always predictable as to which movies will flop and which movies will be successful. However, in the growing film industry, it is imperative to plan ahead and analyse which different factors, strategies and approaches work best in making a movie successful. \n\nThe Box Office Prediction Project aims to solve the above problem by predicting the Box Office performance of a movie based on data such as its cast, director, crew, production company, promotion strategies, release dates, languages, region, plot keywords etc. This Project analyses the performance of previous relevant movies to predict the revenue generated by upcoming movies. The Project aims to learn which strategies and factors are most important in making a movie successful and profitable.')
    df = pd.read_csv("tmdb_5000_movies.csv")
elif button2:
    fig, ax = plt.subplots()
    sns.heatmap(df[['budget', 'month', 'year', 'runtime', 'revenue']].corr(), cmap='YlGnBu', annot=True, linewidths = 0.1)
    plt.title('Correlation of Movie Attributes')

    fig2, ax = plt.subplots()
    sns.heatmap(df[['Cast1_Score', 'Cast2_Score', 'Cast3_Score', 'Cast4_Score', 'Cast5_Score', 'revenue']].corr(), cmap='YlGnBu', annot=True, linewidths = 0.1)
    plt.title('Correlation of Movie Attributes')

    fig3, ax = plt.subplots()
    sns.heatmap(df[['Keywords_Score', 'PC_Score', 'Director_Score', 'Producer_Score', 'revenue']].corr(), cmap='YlGnBu', annot=True, linewidths = 0.1)
    plt.title('Correlation of Movie Attributes')
    
    with st.expander('Heat Maps'):
        st.caption("We've studied the correlation between various movie attributes such as revenue, popularity etc. A positive cell value implies that the row-column pair have a proportional relationship. A negative cell value implies an inversely proportional relationship.")
        st.write(fig)
        st.caption('As seen above, Budget influences Movie Revenue the most, followed by Popularity.')
        st.write(fig2)
        st.write('Here, we examine the relationship between revenue and the Cast.')
        st.write(fig3)
        st.write('Here, we examine the relationship between revenue and the Director, Producer, Production Company and key words')
    with st.expander('Charts'):
        st.empty()
        st.header('')
        st.empty()
        st.header('')
        
        st.caption('Budget heavily influences the Revenue of a Movie.')
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['budget'], df['revenue'], color='RED')
        plt.title('Revenue vs Budget');

        plt.subplot(1, 2, 2)
        plt.bar(df['runtime'], df['revenue'], color='RED')
        plt.title('Revenue vs Runtime');
        plt.xlim(60, 240)        
        
        st.pyplot(plt)
        
        st.empty()
        st.header('')
        st.empty()
        st.header('')
        
        st.caption('Effect of Release Date on Revenue:')
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(df['year'], df['revenue'], color='RED')
        plt.title('Revenue vs Release Year');
        plt.xlim(1900, 2025)
        
        plt.subplot(1, 2, 2)
        plt.bar(df['month'], df['revenue'], width = 1, color='RED')
        plt.title('Revenue vs Release Month');
        
        st.pyplot(plt)
        
        st.empty()
        st.header('')
        st.empty()
        st.header('')
        
        st.caption('Revenue/Budget ratio of various genres: ')
        l = []
        x = genres.copy()
        x.remove('TV Movie')
        for i in x:
            l.append(genres_ratio[i])
        
        plt.subplot(1, 1, 1)
        plt.bar(x, l, width = 1, color='RED')
        plt.title('Revenue vs Genres')
        plt.xticks(rotation=90)
        
        st.pyplot(plt)
        
elif button3:
    with st.expander('Sample Movies'):    
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image('https://www.themoviedb.org/t/p/original/1XXIKzwo8AUQ0KYg9sRwESmNxO8.jpg')
            st.write('Avatar')
            st.caption('Revenue: ' + str(df['revenue'][0]))
            st.caption('Predicted: ' + str(pred[0]))
        with c2:
            st.image('https://www.themoviedb.org/t/p/w500/4ssDuvEDkSArWEdyBl2X5EHvYKU.jpg')
            st.write("Avengers: Age of Ultron")
            st.caption('Revenue: ' + str(df['revenue'][7]))
            st.caption('Predicted: ' + str(pred[7]))
        with c3:
            st.image('https://www.themoviedb.org/t/p/w500/2jLxKF73SAPkyhIWrnv67IH7kJ1.jpg')
            st.write("Spider-Man 3")
            st.caption('Revenue: ' + str(df['revenue'][5]))
            st.caption('Predicted: ' + str(pred[5]))
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image('https://www.themoviedb.org/t/p/original/7GSSyUUgUEXm1rhmiPGSRuKoqnK.jpg')
            st.write('John Carter')
            st.caption('Revenue: ' + str(df['revenue'][4]))
            st.caption('Predicted: ' + str(pred[4]))
        with c2:
            st.image('https://www.themoviedb.org/t/p/original/p3OvQFa5lhbwSAhPygwnlugie1d.jpg')
            st.write("The Lone Ranger")
            st.caption('Revenue: ' + str(df['revenue'][13]))
            st.caption('Predicted: ' + str(pred[13]))
        with c3:
            st.image('https://www.themoviedb.org/t/p/w500/8GFtkImmK0K1VaUChR0n9O61CFU.jpg')
            st.write("Man of Steel")
            st.caption('Revenue: ' + str(df['revenue'][14]))
            st.caption('Predicted: ' + str(pred[14]))    
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image('https://www.themoviedb.org/t/p/original/2YMnBRh8F6fDGCCEIPk9Hb0cEyB.jpg')
            st.write("Pirates of the Caribbean: At World's End")
            st.caption('Revenue: ' + str(df['revenue'][1]))
            st.caption('Predicted: ' + str(pred[1]))
        with c2:
            st.image('https://www.themoviedb.org/t/p/w500/5UsK3grJvtQrtzEgqNlDljJW96w.jpg')
            st.write("Batman v Superman: Dawn of Justice")
            st.caption('Revenue: ' + str(df['revenue'][9]))
            st.caption('Predicted: ' + str(pred[9]))
        with c3:
            st.image('https://www.themoviedb.org/t/p/original/z7uo9zmQdQwU5ZJHFpv2Upl30i1.jpg')
            st.write("Harry Potter and the Half-Blood Prince")
            st.caption('Revenue: ' + str(df['revenue'][8]))
            st.caption('Predicted: ' + str(pred[8]))
    with st.expander('Table'):
        res = new[['title', 'revenue']]
        p = pd.DataFrame(pred, columns = ['Predicted'])
        res = pd.concat([res, p], axis = 1, sort = False)
        res = res.head(25)
        res.rename(columns = {'title':'Title'}, inplace = True)
        res.rename(columns = {'revenue':'Revenue'}, inplace = True)
        st.empty()
        st.header('')
        st.empty()
        st.header('')

        st.caption('A Gradient Boosting Regression ML Algorithm has been applied on a Dataset comprising of 5000 movies. Below are a few sample movies and their corresponding Predicted Revenue.')
        st.table(res)
elif button4:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/fjaoD0ZfPOf2C5BalCziPUaf4Zk.jpg')
        st.write("Avatar 2")
        st.write('')
        st.caption('Predicted: ' + str(upc_pred[0]))
    with c2:
        st.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/blO6k1NsYgPq4sQRZyWvi962FOo.jpg')
        st.write("Jurassic World Dominion")
        st.write('')
        st.caption('Predicted: ' + str(upc_pred[1]))
    with c3:
        st.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/tbUhPhir8TGDkD8RruiBAJE9Nd3.jpg')
        st.write("Lightyear")
        st.write('')
        st.caption('Predicted: ' + str(upc_pred[2]))
elif button5:
    with st.expander('Movies'):
        x = []
        y = []
        ms = pd.read_csv('df1.csv')
        ms = ms.sort_values('revenue', ascending = False)
        for i in range(10):
            x.append(ms.iloc[i]['title'])
            y.append(ms.iloc[i]['revenue'])
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.bar(x, y, width = 0.3, color='RED')
        plt.title('Most Successful Movies')
        plt.xticks(rotation=90)
        
        st.pyplot(plt)
    with st.expander('Genres'):
        i = 0
        x = []
        y = []
        for w in sorted(genres_ratio, key=genres_ratio.get, reverse=True):
            if(genres_count[w] > 30):
                x.append(w)
                y.append(genres_ratio[w])
                i += 1
            if i == 10:
                break
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.bar(x, y, width = 0.3, color='RED')
        plt.title('Most Successful Genres')
        plt.xticks(rotation=90)
        
        st.pyplot(plt)
    with st.expander('Actors'):
        i = 0
        x = []
        y = []
        for w in sorted(actors_ratio, key=actors_ratio.get, reverse=True):
            if(actors_count[w] > 10):
                x.append(w)
                y.append(actors_ratio[w])
                i += 1
            if i == 10:
                break
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.bar(x, y, width = 0.3, color='RED')
        plt.title('Most Successful Actors')
        plt.xticks(rotation=90)
        
        st.pyplot(plt)
    with st.expander('Production Companies'):
        i = 0
        x = []
        y = []
        for w in sorted(pc_ratio, key=pc_ratio.get, reverse=True):
            if(pc_count[w] > 30):
                x.append(w)
                y.append(pc_ratio[w])
                i += 1
            if i == 10:
                break
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.bar(x, y, width = 0.3, color='RED')
        plt.title('Most Successful Production Companies')
        plt.xticks(rotation=90)
        
        st.pyplot(plt)
        
    with st.expander('Producers'):
        i = 0
        x = []
        y = []
        for w in sorted(producers_ratio, key=producers_ratio.get, reverse=True):
            if(producers_count[w] > 10):
                x.append(w)
                y.append(producers_ratio[w])
                i += 1
            if i == 10:
                break
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.bar(x, y, width = 0.3, color='RED')
        plt.title('Most Successful Producers')
        plt.xticks(rotation=90)
        
        st.pyplot(plt)
    with st.expander('Directors'):
        i = 0
        x = []
        y = []
        for w in sorted(directors_ratio, key=directors_ratio.get, reverse=True):
            if(directors_count[w] > 10):
                x.append(w)
                y.append(directors_ratio[w])
                i += 1
            if i == 10:
                break
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.bar(x, y, width = 0.3, color='RED')
        plt.title('Most Successful Directors')
        plt.xticks(rotation=90)
        
        st.pyplot(plt)
    with st.expander('Key Words'):
        i = 0
        x = []
        y = []
        for w in sorted(keywords_ratio, key=keywords_ratio.get, reverse=True):
            if(keywords_count[w] > 30):
                x.append(w)
                y.append(keywords_ratio[w])
                i += 1
            if i == 10:
                break
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.bar(x, y, width = 0.3, color='RED')
        plt.title('Most Successful Key Words')
        plt.xticks(rotation=90)
        
        st.pyplot(plt)
else:
    st.caption('View predicted revenue for any movie!')
    res = new[['title', 'revenue']]
    p = pd.DataFrame(pred, columns = ['Predicted'])
    res = pd.concat([res, p], axis = 1, sort = False)
    entry = st.selectbox('Movie', list(res['title']))
    e = res.loc[res.title == entry]
    x = int(e['revenue'])
    y = int(e['Predicted'])
    st.info('Actual Revenue: ' + str(x))
    st.info('Predicted Revenue: ' + str(y))
    
    st.empty()
    st.header('')
    st.empty()
    st.header('')
    
    st.caption('Use your imagination and test out the BOXOP Algorithm!')
    
    title = st.text_input('Movie Title', 'Example')
    budget = st.number_input('Budget', 100000000)
    runtime = st.number_input('Runtime', 120)
    month = st.number_input('Month of Release', 6)
    year = st.number_input('Year of Release', 2022)
    genre = st.multiselect('Genres', genres)
    pc = st.selectbox('Production Company', list(pc_ratio.keys()))
    pro = st.selectbox('Producer', list(producers_ratio.keys()))
    dire = st.selectbox('Director', list(directors_ratio.keys()))
    keywords = st.multiselect('Key Words', list(keywords_ratio.keys()))
    cast = st.multiselect('Cast', list(actors_ratio.keys()))
    
    lim = min(5, len(cast))
    cast = cast[:lim]
    lim = min(10, len(keywords))
    keywords = keywords[:lim]
    custom = []
    movie = []
    movie.append([budget, runtime, month, year, genres, cast, keywords, pc, pro, dire])
    for i in movie:
        m = []
        m.append(i[0])
        m.append(i[1])
        m.append(i[2])
        m.append(i[3])
        for j in genres:
            if j in i[4]:
                m.append(1)
            else:
                m.append(0)            
        for k in i[5]:
            if j in actors_ratio:
                m.append(actors_ratio[j])
            else:
                m.append(0)
        for k in range(5 - len(i[5])):
            m.append(0)
        for j in i[6]:
            x = 0
            c = 0
            if j in keywords_ratio:
                x += keywords_ratio[j]
                c += 1
        m.append((x / c) * 10)
        if i[7] in pc_ratio:
            m.append(pc_ratio[i[7]])
        else:
            m.append(0)
        if i[8] in producers_ratio:
            m.append(producers_ratio[i[8]])
        else:
            m.append(0)
        if i[9] in directors_ratio:
            m.append(directors_ratio[i[9]])
        else:
            m.append(0)
        custom.append(m)
    cus_pred = reg.predict(custom)
    st.write('Predicted Revenue:')
    st.info(str(cus_pred[0]))