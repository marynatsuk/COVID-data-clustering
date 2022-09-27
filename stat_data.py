import pandas as pd

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

corona_dataset_clear = corona_dataset_csv.drop(lst_columns, axis=1)
corona_dataset_clear.set_index('location', inplace=True)
corona_dataset_clear.drop(lst_rows, axis=0, inplace=True)

corona_dataset_aggregated = corona_dataset_clear.groupby(['location']).max()
corona_dataset_aggregated = corona_dataset_aggregated[corona_dataset_aggregated['total_cases_per_million'].notna()]
corona_dataset_aggregated = corona_dataset_aggregated[corona_dataset_aggregated['total_deaths_per_million'].notna()]

#=============================================================
world_happiness_csv = pd.read_csv("worldwide_happiness_report.csv")
useless_columns = ['Overall rank', 'Generosity', 'Perceptions of corruption', 'Score', 'Healthy life expectancy']
happiness_aggregated = world_happiness_csv.drop(useless_columns, axis=1)
happiness_aggregated.set_index('Country or region', inplace=True)

#=================================================================
df_plus_happiness = corona_dataset_aggregated.join(happiness_aggregated, how='inner')
df_plus_happiness = df_plus_happiness.fillna(df_plus_happiness.mean())
print(list(df_plus_happiness.columns))
df_plus_happiness.reset_index(drop=True, inplace=True)
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

print(df_plus_happiness.mean())






