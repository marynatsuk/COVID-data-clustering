import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from kneed import KneeLocator
from collections import Counter

#----------- Step 1. Обробка вхідного датасету -----------------------
corona_dataset_csv = pd.read_csv("owid-covid-data.csv", header=[0])

lst_columns = ['continent', 'total_cases', 'total_deaths', 'date', 'iso_code', 'new_cases', 'new_cases_smoothed','new_deaths', 'new_deaths_smoothed',
               'new_cases_per_million', 'new_cases_smoothed_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million',
               'icu_patients', 'icu_patients_per_million', 'hosp_patients', 'hosp_patients_per_million',
               'weekly_icu_admissions', 'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
               'weekly_hosp_admissions_per_million', 'total_tests', 'new_tests', 'total_tests_per_thousand',
               'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
               'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations', 'people_vaccinated',
               'people_fully_vaccinated', 'total_boosters', 'new_vaccinations', 'new_vaccinations_smoothed',
               'total_vaccinations_per_hundred',  'people_fully_vaccinated_per_hundred',
               'total_boosters_per_hundred', 'new_vaccinations_smoothed_per_million', 'new_people_vaccinated_smoothed',
               'new_people_vaccinated_smoothed_per_hundred', 'population', 'aged_65_older', 'aged_70_older',
               'gdp_per_capita', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers',
               'handwashing_facilities', 'hospital_beds_per_thousand', 'excess_mortality_cumulative_absolute',
               'excess_mortality_cumulative', 'excess_mortality', 'excess_mortality_cumulative_per_million',
               'people_vaccinated_per_hundred', 'extreme_poverty']
lst_rows = ['Africa', 'Europe', 'Upper middle income', 'South America', 'World', 'North America', 'Asia', 'North Korea']

#очищуємо датасет від неподрібних колонок та строк
corona_dataset_clear = corona_dataset_csv.drop(lst_columns, axis=1)
corona_dataset_clear.set_index('location', inplace=True)
corona_dataset_clear.drop(lst_rows, axis=0, inplace=True)

corona_dataset_aggregated = corona_dataset_clear.groupby(['location']).max()
corona_dataset_aggregated = corona_dataset_aggregated[corona_dataset_aggregated['total_cases_per_million'].notna()]
corona_dataset_aggregated = corona_dataset_aggregated[corona_dataset_aggregated['total_deaths_per_million'].notna()]
corona_dataset_aggregated.to_csv("output.csv")

#----------- Step 2. Обробка другого датасету ------------------------------
world_happiness_csv = pd.read_csv("worldwide_happiness_report.csv")
useless_columns = ['Overall rank', 'Generosity', 'Perceptions of corruption', 'Score', 'Healthy life expectancy']
happiness_aggregated = world_happiness_csv.drop(useless_columns, axis=1)
happiness_aggregated.set_index('Country or region', inplace=True)

#----------- Step 3. З'єднання двох датасетів -------------------------------
df_plus_happiness = corona_dataset_aggregated.join(happiness_aggregated, how='inner')
df_plus_happiness = df_plus_happiness.fillna(df_plus_happiness.mean())
# df_plus_happiness.reset_index(inplace=True)
# df_plus_happiness.rename(columns={'index':'location'}, inplace=True)
df_plus_happiness.rename(columns = {'total_cases_per_million':'cases_per_mil',
                                    'total_deaths_per_million':'deaths_per_mil',
                                    'reproduction_rate':'reprod_rate',
                                    'stringency_index':'string_index',
                                    'population_density':'pop_density',
                                    'life_expectancy':'life_expect',
                                    'human_development_index':'hum_dev_index',
                                    'GDP per capita':'gdp_per_capita',
                                    'Social support':'social_support',
                                    'Freedom to make life choices':'life_choices'}, inplace=True)
df_plus_happiness.to_csv("output_final.csv")

#----------- Кореляції в датасеті --------------------------------------------

corrmat = df_plus_happiness.corr()
top_corr_features = corrmat.index
fig = sns.heatmap(df_plus_happiness[top_corr_features].corr(),annot=True,cmap="RdYlGn")
fig.figure.tight_layout()
plt.show()

#------------ Кластеризація -------------------------------------------------

def pca(dataset):
    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(dataset)
    return pca_2_result

def tsne(dataset):
    tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(dataset)
    # cluster.tsneplot(score=tsne_em)
    # print(tsne_em)
    return tsne_em

def kneelocator(dataset):
    neighbors = NearestNeighbors(n_neighbors=5).fit(dataset)
    distance, index = neighbors.kneighbors(dataset)
    distance = np.sort(distance[:, 4], axis=0)
    i = np.arange(len(distance))
    knee = KneeLocator(i, distance, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    knee.plot_knee()
    plt.show()

    result = distance[knee.knee]
    print(distance[knee.knee])
    return result

def dbscan_clustering(dataset, knee):
    clusters = DBSCAN(eps=0.8, min_samples=17).fit(dataset)
    print(clusters.labels_)
    print(Counter(clusters.labels_))
    labels = clusters.labels_

    sns.scatterplot(data=dataset, x=dataset[:, 0], y=dataset[:, 1], hue=labels, palette=['blue', 'coral', 'green', 'purple'], legend='full')
    plt.title('DBSCAN clustering')
    plt.show()

def kmeans_clustering(dataset):
    result = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++')
        kmeans.fit(dataset)
        result.append(kmeans.inertia_)
    plt.plot(range(1, 11), result)
    plt.title('Elbow method')
    plt.show()

    kmeans = KMeans(n_clusters=4)
    label = kmeans.fit_predict(dataset)
    print(label)
    print(Counter(label))
    sns.scatterplot(data = dataset, x=dataset[:, 0], y=dataset[:, 1], hue=label, palette=['blue', 'coral', 'green', 'purple'], legend='full')
    plt.title('KMeans clustering')
    plt.show()
    return label

def hierarchical_clustering(dataset):
    plt.figure(figsize=(10, 7))
    plt.title("Dendogram")
    shc.dendrogram(shc.linkage(dataset, method='ward'))
    plt.show()
    clusters = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    label = clusters.fit_predict(dataset)
    sns.scatterplot(data=dataset, x=dataset[:, 0], y=dataset[:, 1], hue=clusters.labels_,
                palette=['blue', 'coral', 'green', 'purple'], legend='full', s=50)
    plt.title('Hierarchical clustering')
    plt.show()
    print(clusters.labels_)
    print(Counter(clusters.labels_))
    return label

if __name__=="__main__":
    df_plus_happiness[df_plus_happiness.columns] = StandardScaler().fit_transform(df_plus_happiness)
    df_plus_happiness.to_csv('df_standard.csv')
    #DBSCAN PCA
    print('DBSCAN clustering')
    pca_dataset_dbscan = pca(df_plus_happiness)
    knee_tsne = kneelocator(pca_dataset_dbscan)
    dbscan_clustering(pca_dataset_dbscan, knee_tsne)
    #KMEANS PCA
    print('KMEANS clustering')
    pca_dataset_kmeans = pca(df_plus_happiness)
    labels_pca = kmeans_clustering(pca_dataset_kmeans)
    #HCA PCA
    print('Hierarchical clustering')
    pca_dataset_hca = pca(df_plus_happiness)
    labels_hca = hierarchical_clustering(pca_dataset_hca)



