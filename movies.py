import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
pd.options.plotting.backend = "plotly"
pd.set_option('display.max_columns', None)


class Rec:
    def __init__(self,visitor_id,proc_type='k3',make_test_data=False):
        self.visitor_id = visitor_id
        self.proc_type = proc_type
        self.movies_pre()#обрабатываем файл в подходящий нам DtFr
        if make_test_data:
            self.make_test_visitor_views()
            self.make_test_rating(self.movies)
        self.visitor_pref_actors=self.get_visitor_actors()
        self.set_visitor_watched()
        self.remove_watched_from_movies()
        self.set_main_rating_list()

    def movies_pre(self):
        movies = pd.read_csv('C:/Users/maili/Downloads/movies.csv',
                     usecols=['names', 'date_x', 'score', 'genre', 'orig_title', 'status', 'crew', 'orig_lang', 'budget_x',
                              'country'])# используем не все нп revenue #dtype={'score': 'Int32', 'budget_x': 'Int32'} parse_dates=['date_x']
        movies['date_x'] = pd.to_datetime(movies['date_x'])#.dt.strftime("%Y-%m-%d")#американские даты в нормальный формат
        #print(movies.head())        #print(movies.info())        #exit()
        movies.index = [x for x in range(1, len(movies.index) + 1)]# добавляем индекс с единицы
        movies.index.name = 'id'
        movies.drop_duplicates(subset=["orig_title", "date_x","country"], keep='first',inplace=True)#убираем дублирующиеся фильмы
        temp = movies.select_dtypes(['object'])#отбираем колонки для trim strings
        movies[temp.columns] = temp.apply(lambda x: x.str.strip())
        #print(movies.columns.values)        # print(movies.loc[16])
        movies['freq'] = movies.groupby('country')['country'].transform('count')  # lambda x: x.count()/10178
        # print(movies.loc[16])        # print(movies.head())
        # remove NON Released, and without rating,budget
        movies = movies[(movies['status'] == 'Released') & (movies['score'] > 0) & (movies['budget_x'] > 0)]
        # remove nan
        movies.dropna(inplace=True)
        movies['genre'] = movies['genre'].str.replace(' ', '_')
        del movies['status']
        self.movies=movies.copy()
        self.movies['mtype'] = self.movies.apply(lambda x: self.movie_type(x['budget_x'], x['country'], x.name), axis=1)#уровень бюджета фильма
        self.movies_full = self.movies.copy()

    def movie_type(self,budget,country,ind):
            c1=['AU','US']
            c2=['DE','CN','IT','FR','IN','GB','CA','ES']
            if country in c1:
                if budget>70000000:
                    mtype='A'
                elif budget>30000000:
                    mtype='B'
                elif budget>10000000:
                    mtype='C'
                elif budget>2000000:
                    mtype='D'
                else:
                    mtype='E'
            elif country in c2:
                if budget>70000000:
                    self.movies.loc[ind,'budget_x']=70000000
                    mtype='A'
                elif budget>30000000:
                    mtype='A'
                elif budget>10000000:
                    mtype='B'
                elif budget>5000000:
                    mtype='C'
                elif budget>2000000:
                    mtype='D'
                else:
                    mtype='E'
            else:
                if budget>50000000:
                    self.movies.loc[ind,'budget_x']=50000000
                    mtype='A'
                elif budget>20000000:
                    mtype='A'
                elif budget>5000000:
                    mtype='B'
                elif budget>3000000:
                    mtype='C'
                elif budget>1000000:
                    mtype='D'
                else:
                    mtype='E'
            return mtype

    def make_test_visitor_views(self):
        #формируем тестовую историю просмотров тестового посетителя
        self.visitor_watched = {293: 5, 2: 7, 6: 0, 9: 0, 7: 0, 148: 7, 219: 5, 360: 8, 5677: 8, 8607: 0, 8839: 0, 3416: 0, 3399: 7,
                   28: 7, 30: 7, 69: 6, 92: 6, 132: 6, 330: 7, 356: 7, 363: 7, 357: 0, 373: 7, 389: 3, 394: 7, 432: 3,
                   506: 8, 524: 8, 529: 0, 548: 7, 592: 7, 598: 8, 1494: 9, 1491: 8, 1513: 0, 1522: 0, 1529: 9, 1521: 7,
                   1563: 0, 1567: 0, 1568: 0, 1569: 0, 4348: 0, 6827: 6, 1617: 7, 1676: 8, 8062: 0, 8055: 8, 8739: 0,
                   181: 8, 1787: 7, 1786: 4, 1925: 7, 1928: 0, 1943: 0, 2123: 8, 2130: 6, 5478: 9, 254: 5, 2444: 8,
                   2574: 7, 1799: 0, 9956: 6}

        df_watched = pd.DataFrame.from_dict(data=self.visitor_watched, orient='index').reset_index()
        df_watched.columns = ['id', 'rating']
        df_watched['w_date'] = pd.date_range(start='2020-03-09', end='2023-05-02', periods=df_watched.shape[0],
                                             normalize=True)
        df_watched.to_csv("C:/Users/maili/Downloads/movies_watched.csv")
    def make_test_rating(self,movies):
        # формируем тестовые оценки других пользователей - около 200 человек с оценками от 10 до 60 фильмов
        movies['age'] = ((pd.to_datetime("now") - movies.date_x) / np.timedelta64(1, 'Y'))
        movies['age'] = movies['age'].astype(int)
        # print(movies.tail())       # print(movies['age'].min())        # exit()
        countries = ['AU', 'RU', 'US', 'GB', 'KR', 'HK', 'JP', 'IN', 'FR', 'ES', 'IT', 'ME', 'DE', 'CN', 'CA', 'AR',
                     'BR']
        c_pref_def = dict.fromkeys(countries, 3)
        # группы: любители индийских фильмов, #любители американских фильмов, #любители российских фильмов, #любители азиатских фильмов, #смешаная группа
        # заменяем значения по умолчанию
        c_pref = [c_pref_def | {'RU': 50, 'SU': 50, 'IN': 100}, c_pref_def | {'RU': 40, 'US': 100, 'AU': 100},
                  c_pref_def | {'RU': 100, 'SU': 50, 'US': 30, 'AU': 50},
                  c_pref_def | {'RU': 20, 'US': 20, 'AU': 20, 'HR': 80, 'KR': 100, 'JP': 80},
                  c_pref_def | {'RU': 40, 'SU': 30, 'US': 40, 'AU': 50, 'KR': 15, 'IN': 3, 'FR': 15, 'ES': 15, 'IT': 15,
                                'CN': 20, 'CA': 15}]
        # print(c_prob)
        rating_list = dict()
        visitor_watched = self.visitor_watched.keys()
        # увеличиваем шанс просмотра фильмов пользователя, чтобы потом попались похожие пользователи
        for el in visitor_watched:
            movies.loc[el, 'freq'] = round(movies.loc[el, 'freq'] / 1.3)
        user_id = 1
        for i in range(5):
            memb = rnd.randrange(18, 50, 5)  # количество человек в группе
            # ранжировка любимых стран группы
            movies['pref'] = 1  # добавляем колонку с дефолтным значением вероятности отбора фильма
            # берём любимые страны группы
            for key, value in c_pref[i].items():
                movies.loc[movies['country'] == key, 'pref'] = value

            movies['prob'] = movies['pref'] * abs(movies['score'] - movies['age'] / 2.5) * 30 / movies[
                'freq']  # предпочтение, рейтинг чем выше тем больше смотрят - возвраст, чем старше тем меньше смотрят/ и делить на частоту чтобы уменьшить её влияние
            for j in range(memb):
                user_id += 1
                movnum = rnd.randint(8, 38)  # количество оценённых фильмов человеком
                minrate = 10 - round(2 + 0.11 * movnum)  # минимальный рейтинг, чем меньше фильмов, тем ставит выше
                weighted_sample = movies.sample(n=movnum, weights="prob")#отбираем используя вероятность
                weighted_sample['rating'] = np.random.randint(minrate, 11, weighted_sample.shape[0])
                rating_list[user_id] = weighted_sample
                # print(weighted_sample[["score","country",'age','rating']])  # exit()

        rating_list = pd.concat(rating_list.values(), keys=rating_list.keys(), names=['user_id'])
        rating_list.drop(
            ['names', 'date_x', 'score', 'genre', 'crew', 'orig_title', 'orig_lang', 'budget_x', 'country', 'mtype',
             'freq', 'age', 'pref', 'prob'], axis=1, inplace=True)
        rating_list.to_csv("C:/Users/maili/Downloads/movies_rating3.csv")

    def get_visitor_actors(self):
        df_pref_actors = {0:{'Chris Pratt', 'Keanu Reeves', 'Gerard Butler', 'Dwayne Johnson'}}
        return df_pref_actors[self.visitor_id]

    def set_visitor_watched(self):
        self.visitor_watched_base = pd.read_csv('C:/Users/maili/Downloads/movies_watched.csv', index_col=0)#self.visitor_id
        self.visitor_watched_ids=self.visitor_watched_base['id'].to_numpy()
        # высчитываем что он больше всего смотрит: жанры, страны
        self.df_visitor_watched = pd.merge(self.visitor_watched_base, self.movies[["names", "genre", "country"]], on="id", how="left")
        nstr2 = self.df_visitor_watched['genre'].str.split(',\xa0').to_numpy()
        nstr2 = np.concatenate(nstr2, axis=0)
        g, c = np.unique(nstr2, return_counts=True)  # считаем частоту
        genres_count = list(zip(g, c / sum(c)))  # закидываем с частотой в процентах
        self.genres_count = dict(sorted(genres_count, key=lambda item: item[1], reverse=True)) # закидываем в словарь с сортингом по ключу
        self.counties_pref = dict(self.df_visitor_watched['country'].value_counts(normalize=True))
        # print(counties_pref)
        # plt.bar(counties_pref.keys(),counties_pref.values())
        # plt.show()
    def set_main_rating_list(self):
        self.rating_list = pd.read_csv("C:/Users/maili/Downloads/movies_rating3.csv",
                                  usecols=['user_id', 'id', 'rating'])  # список оценок другими прользователями

    def remove_watched_from_movies(self):
        self.movies = self.movies[~self.movies.index.isin(self.visitor_watched_ids, level='id')].copy()  # убираем просмотренные визитором
        # print('after:',movies.shape[0])

    def out_print(self,bmovies,title):
        print('\n'+title+':\n')
        for i, row in bmovies.iterrows():
            print(f">: {row['names']} ({row['orig_title']}, {row['country']}, {row['date_x'].strftime('%Y/%m/%d')})")

    def show_big(self):
        # 5 новых больших фильма, можно просто по дате - 10 последних + shuffle (bmovies.sort_values(by='date_x', ascending=False)), но беруём исходя из страновых предпочтений
        bmovies = self.movies[
            (self.movies['mtype'] == 'A') & (self.movies['date_x'] > pd.to_datetime("now") - pd.Timedelta(weeks=32))].copy()
        bmovies['freq'] = bmovies.groupby('country')['country'].transform('count')
        bmovies['pref'] = 1
        for key, value in self.counties_pref.items():
            value = value * 100
            if value > 70:
                value = 70
            elif value < 4:
                continue
            bmovies.loc[bmovies['country'] == key, 'pref'] = np.ceil(value)

        bmovies['prob'] = np.ceil(
            bmovies['pref'] * bmovies['score'] * (bmovies['budget_x'] / 20000000) / bmovies['freq'])
        bmovies = bmovies.sample(n=5, weights="prob")
        self.out_print(bmovies,'Big Movies set')

    def show_new(self):
        # 5 новинки с хорошим рейтингом ()
        bmovies = self.movies[
            (self.movies['mtype'].isin(['B', 'C'])) & (self.movies['date_x'] > pd.to_datetime("now") - pd.Timedelta(weeks=32)) & (
                        self.movies['score'] > 70)].copy()
        bmovies.sort_values(by='date_x', ascending=False, inplace=True)
        bmovies = bmovies.head(5)
        self.out_print(bmovies,'New Movies set')

    def show_similar(self):
        # 5 похожих которые вам понравились фильма с лучшим рейтингом
        bmovies = self.df_visitor_watched[0:50].sort_values(by='rating', ascending=False)  #отбираем последние просмотренные
        bmovies = bmovies[['genre', 'country']][0:5]
        bmovies = self.movies.merge(bmovies, how='inner', on=['genre', 'country'])#отбираем из базы такие же по стране и жанру
        bmovies.sort_values(by='score', ascending=False, inplace=True)
        bmovies = bmovies[0:5]
        self.out_print(bmovies,'Similar Movies set')


    def show_same_users(self):
        # 5 фильмов от людей, которые оценили также как и visitor максимальное количество фильмов (можно за минусом фильмов с противоположным мнением)
        union = self.rating_list.merge(self.visitor_watched_base, how="inner",
                                  on=['id', 'rating'])  # оставляем только общие рейтинги
        union_users = union.groupby('user_id')['user_id'].count().nlargest(5)  # самые совпавшие товарищи
        union_users = union_users.index.get_level_values(0).tolist()  # конвертируем их в список
        self.rating_list5 = self.rating_list[self.rating_list['user_id'].isin(union_users) & (self.rating_list['rating'] > 7) & (
            ~self.rating_list['id'].isin(self.visitor_watched_base['id']))].sample(5)  # отбираем 5 случайных с нормальным рейтингом от похожих
        bmovies = pd.merge(self.rating_list5, self.movies, on='id', how='inner')  # дополняем отобраный список фильмов информацией
        self.out_print(bmovies,'Same users set')

    def g_pref(self,xcell):
        pref = 0
        for g, c in self.genres_count.items():
            if xcell.find(g) != -1:
                pref += round(c * 100)  # для проверки g+':'+round(c*100).__str__()+'|'
        return pref
    def show_in_genres(self):
        # 5 фильма из категорий которые нравятся пользователю
        bmovies = self.movies[self.movies['score'] > 70].copy()
        bmovies['genre_pref'] = 0
        bmovies['genre_pref'] = bmovies.apply(lambda x: self.g_pref(x['genre']),
                                              axis=1)  # добавляем рейтинг для visitor по жанрам
        bmovies.sort_values(by='genre_pref', ascending=False, inplace=True)
        bmovies = bmovies[0:30].sample(n=5)
        self.out_print(bmovies,'By prefered genres set')

    def show_old(self):
        # 5 фильмов старой классики
        bmovies = self.movies[
            (self.movies['mtype'].isin(['A', 'B'])) & (self.movies['date_x'] < '1991-12-12') & (self.movies['score'] > 73)].copy()
        bmovies['freq'] = bmovies.groupby('country')['country'].transform('count')
        bmovies['pref'] = 1
        for key, value in self.counties_pref.items():
            value = value * 100
            if value > 70:
                value = 70
            elif value < 4:
                continue
            bmovies.loc[bmovies['country'] == key, 'pref'] = np.ceil(value)

        bmovies['prob'] = np.ceil(bmovies['pref'] / bmovies['freq'])
        bmovies = bmovies.sample(n=5, weights="prob")
        bmovies = bmovies.sample(n=5)  # print(bmovies)
        self.out_print(bmovies,'Old school set')

    def show_in_actors(self):
        # 5 фильмов с любимыми актёрами
        bmovies = self.movies[self.movies['crew'].str.contains("|".join(self.visitor_pref_actors))].copy()
        bmovies['age'] = ((pd.to_datetime("now") - bmovies.date_x) / np.timedelta64(1, 'Y'))
        bmovies['age'] = bmovies['age'].astype(int)
        bmovies.sort_values(['age', 'score'], ascending=[True, False], inplace=True)
        bmovies = bmovies[0:5]
        self.out_print(bmovies, 'Actors set')

    def show_in_cluster(self):
        # 5 фильмов из кластера визитёра
        visitor_watched_base = self.visitor_watched_base[['id', 'rating']]
        visitor_watched_base = visitor_watched_base[visitor_watched_base['rating'] > 0]
        visitor_watched_base['user_id'] = 0
        data_temp = pd.concat([self.rating_list, visitor_watched_base],
                              ignore_index=True)  # добавляем визитера к оценкам других пользователей
        data_temp = pd.merge(data_temp, self.movies_full[['names', 'orig_title', 'date_x', 'genre', 'country']], how='left',
                             on='id', copy=True).copy()  # добавляем в оценцки информацию о фильмах#movies_full, тк из movies удалены просмотренные
        data_temp.dropna(inplace=True)
        data_temp_orig = data_temp.copy()#делаем копию оригинальных данных без стандартизации и перевода в количественные

        # переводим качественные данные в количественные
        genre_str = data_temp['genre'].to_string(header=False, index=False)
        genre_str = '/'.join(genre_str.splitlines())
        genre_str = genre_str.replace(" ", "")
        genre_str = genre_str.replace(",\xa0", "/")
        genres = set(genre_str.split('/'))
        for genre in genres:
            data_temp[genre] = np.where(data_temp['genre'].str.contains(genre), 1, 0)#добавляем столбцы с жанрами нп onehot

        # убираем id и тайтлы, разбиваем дату на периоды, переводим страны в колонки нп onehot
        countries_cols = np.unique(data_temp["country"].to_numpy())
        for ccol in countries_cols:
            data_temp[ccol] = np.where(data_temp['country'] == ccol, 1, 0)  # lambda a : 1 if(a=1 ) else 0
        bins = [pd.to_datetime('1900-01-01'), pd.to_datetime('1981-01-01'), pd.to_datetime('2001-01-01'),
                pd.to_datetime('2023-09-01')]
        dlabels = ['C', 'B', 'A']
        # Даты на бины
        data_temp['date_bin'] = pd.cut(data_temp['date_x'], bins, labels=dlabels)
        for bcol in dlabels:
            data_temp[bcol] = np.where(data_temp['date_bin'] == bcol, 1, 0)
        data_temp = data_temp.copy()
        # убираем малозначимые страны в одну колонку - EX
        countries_cols_curr = list()
        data_temp['EX'] = 0
        for column in countries_cols:
            if (data_temp[column].mean() < 0.01):
                data_temp['EX'] += data_temp[column]
                data_temp.drop(column, inplace=True, axis=1)
            else:
                countries_cols_curr.append(column)
        countries_cols_curr.append('EX')
        # print(data_temp.describe())
        cols = np.delete(data_temp.columns, [range(8)], axis=0)
        cols = cols[cols != 'date_bin']
        # print(cols)
        grouped_data = data_temp.groupby('user_id')[cols].sum()#групируем по юзерам и подсчитываем какие жанры и эпоху предпочитают

        genres = list(genres)
        genres_max = grouped_data[genres].max().max()  # максимальное количество в колонке жанры
        countries_max = grouped_data[countries_cols_curr].max().max()  # максимальное количество в колонке страны
        dlabels_max = grouped_data[dlabels].max().max()  # максимальное число в эпохах
        # print(genres_max,countries_max,dlabels_max)
        data_frac = grouped_data.copy()  # делаем вариант с долями/процентами/пропорцией
        data_frac[genres] = round(data_frac[genres] / genres_max, 4)
        data_frac[countries_cols_curr] = round(data_frac[countries_cols_curr] / countries_max, 4)
        data_frac[dlabels] = round(data_frac[dlabels] / dlabels_max, 4)
        # exit(print(data_frac))
        data_norm = grouped_data.copy()# делаем вариант с стандартизацией/нормализацией
        scaler = preprocessing.RobustScaler()  # StandardScaler()#MinMaxScaler()#RobustScaler()
        data_norm = scaler.fit_transform(data_norm)

        if self.proc_type=='kr':
            #вариант кластеризации с kmeans и полной размерностью датафрейма
            clusters=self.kmeans_raw(data_frac)
        else:
            #self.proc_type == 'k3':
            #вариант кластеризации с kmeans и трехмерной размерностью датафрейма
            clusters=self.kmeans_3d(data_norm,show3d=False)
        #вариант с hdbscan не давал нормальных результатов import hdbscan # clast=hdbscan.HDBSCAN().fit_predict(data_frac)

        # получаем кластер визитера
        visitor_cluster = clusters[0]
        # присваиваем пользователям кластеры
        users_cl = pd.DataFrame(list(zip(grouped_data.index.get_level_values(0), clusters)), columns=['user_id', 'clust'])

        # добавляем в оценки фильмов кластеры
        rating_list_cl = pd.merge(data_temp_orig, users_cl, how='inner', on='user_id')

        if users_cl['clust'].value_counts()[visitor_cluster] < 2:
            visitor_cluster = users_cl[
                'clust'].value_counts().idxmax()  # если visitor один в своём кластере, то присваиваем ему самый популярный кластер

        # оставляем только фильмы с хорошими оценками из кластера за вычетом тех, которые получились при отборе у пользователей с совпавшими оценками
        bmovies = rating_list_cl[
            (rating_list_cl['clust'] == visitor_cluster) & (~rating_list_cl['id'].isin(visitor_watched_base['id'])) & (
                ~rating_list_cl['id'].isin(self.rating_list5['id'])) & (rating_list_cl['rating'] > 7)]
        if bmovies.shape[0] > 4:
            bmovies = bmovies.sample(5)
        self.out_print(bmovies,'From cluster set')


    def kmeans_raw(self,data):
        # определяем количество кластеров Методом локтя
        ''''
        sqerr_l=list()
        for i in range(2,30):
            km=KMeans(n_clusters=i,n_init='auto').fit(data_frac)
            sqerr_l.append(km.inertia_)
        plt.plot(range(2,30),sqerr_l)
        plt.show()
        '''
        # 12 кластеров, и с долями
        kmeans = KMeans(12, n_init='auto')
        clusters = kmeans.fit_predict(data)
        return clusters

    def kmeans_3d(self, data, show3d=False):
        d2 = PCA(n_components=3).fit_transform(data)#редуцируем до 3х компонент
        d2 = pd.DataFrame(d2)
        # sqerr_l=list()
        # for i in range(2,30):
        #     km=KMeans(n_clusters=i,n_init='auto').fit(d2)
        #     sqerr_l.append(km.inertia_)
        # plt.plot(range(2,30),sqerr_l)
        # plt.show()
        # 8 кластеров
        # exit()
        kmeans = KMeans(8, n_init='auto')
        clusters = kmeans.fit_predict(d2)
        if show3d:
            self.plot_cluster3d(d2, clusters)

        return clusters

    def plot_cluster3d(self, X, y, title="Viewers 3D scatter"):
        fig = plt.figure()
        fig = fig.add_subplot(projection='3d')
        fig.scatter(X[0], X[1], X[2], c=y)
        fig.scatter(X.iloc[0,0], X.iloc[0,1], X.iloc[0,2], c='r', s=50)#наш visitor
        fig.set_xlabel('X Label')
        fig.set_ylabel('Y Label')
        fig.set_zlabel('Z Label')
        fig.set_title(title)
        plt.show()


recommend = Rec(0)

recommend.show_big()#крупные премьеры
recommend.show_new()#новые подходящие фильмы
recommend.show_similar()#похожие на просмотренные
recommend.show_same_users()#от других оценивших как визитор
recommend.show_in_genres()#подходящие по жанрам
recommend.show_old()#старые фильмы
recommend.show_in_actors()#с любимыми актерами
recommend.show_in_cluster()#подходящие из кластера
