# %%
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os


# %% Use API for geet
def social_variables():
    social_variable = {
        "GEO_ID": "AFFGEOID",
        "DP05_0001E": "Total population",
        "DP05_0019PE": "Population under 18",
        "DP05_0024PE": "Population over 65",
        "DP05_0065PE": "Black or African",
        "DP05_0071PE": "Hispanic or Latino",
        "DP02_0114PE": "Non English speaker",
        "DP02_0154PE": "With internet",
        "DP02_0153PE": "With computer",
        "DP02_0067PE": "High school or Over",
        "DP03_0099PE": "Without health insurance",
        "DP02_0072PE": "With disability",
        "DP03_0088E": "Per capita Income",
        "DP03_0119PE": "Under poverty",
        "DP03_0062E": "Median household income",
        "DP04_0080E": "Median house value",
        "DP03_0004PE": "Percent of employment",
    }

    return social_variable


def tract_data(year, path, state_code):
    census_file = f"{path}/censusdata.pkl"
    social_variable = social_variables()

    if os.path.isfile(census_file):
        print("read cached file")
        census_dataframe = pd.read_pickle(census_file)

    else:
        my_key = "1643018d80128c7282a2297e09b2f6aba72806f9"
        census_data = requests.get(
            f"https://api.census.gov/data/{year}/acs/acs5/profile?get={','.join(list(social_variable.keys()))}&for=tract:*&in=state:{state_code}&key={my_key}"
        )

        poverty_percent_county = census_data.json()
        columns = poverty_percent_county[0]
        census_dataframe = pd.DataFrame(
            columns=columns, data=poverty_percent_county[1:]
        )
        print(f"There are {census_dataframe.shape[0]} in original dataset")
        census_dataframe = census_dataframe.rename(columns=social_variable)
        non_str_columns = list(social_variable.values())[1:]
        census_dataframe[non_str_columns] = census_dataframe[non_str_columns].astype(
            float
        )
        census_dataframe = census_dataframe.loc[
            ~(census_dataframe[non_str_columns] < 0).any(axis=1)
        ]
        print(f"After cleaning, there are {census_dataframe.shape[0]} remaining")

        census_dataframe[["tract", "county"]] = (
            census_dataframe[["tract", "county"]].astype(int).astype(str)
        )
        census_dataframe.to_pickle(census_file)

    return census_dataframe


if __name__ == "__main__":
    os.chdir("/Users/xudongfan/ResearchProjects/PROJ-2023-MLSERN/PowerSystemCase")

    tt = tract_data(2021)
