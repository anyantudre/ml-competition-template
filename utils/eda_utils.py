import numpy as np
import pandas as pd
from scipy import stats
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.pyplot as plt

from src import config

# function to import our dataset
def import_data(train_path, test_path, submission_path):
    train       =  pd.read_csv(train_path)
    test        =  pd.read_csv(test_path)
    submission  =  pd.read_csv(submission_path)
    return train, test, submission


def print_first_rows(train, test, submission, n_rows=3):
    print(f"First few rows training set:")
    print("==="*8)
    print(train.head(n_rows).to_string(index=False))
    print("\n")

    print(f"First few rows test set:")
    print("==="*8)
    print(test.head(n_rows).to_string(index=False))
    print("\n")

    print(f"First few rows Submission:")
    print("==="*8)
    print(submission.head(n_rows).to_string(index=False))
    print("\n")


def print_shapes(train, test, submission):
    print(f"Shape of the data:")
    print("==="*6)
    print(f"""Training: {train.shape} \nTest: {test.shape} \nSubmission: {submission.shape}""")
    print("\n")


def print_duplicated(train, test, submission):
    print(f"Number of duplicated values:")
    print("==="*9)
    print(f"""Training: {train.duplicated().sum()} \nTest: {test.duplicated().sum()} \nSubmission: {train.duplicated().sum()}""")
    print("\n")


def print_missing(df):
    print(f"Number of missing values by cols in {df}:")
    print("==="*15)
    print(f"""{df.isna().sum().sort_values(ascending=False)}""")
    print("\n")

def print_value_counts(df, cols):
    print("Number of values in each categorical column:")
    print("==" * 15)
    for col in cols:
        print(f"{df[col].value_counts()}")
        print("==" * 15)
    print("\n")


# fonction to calculate univariate stats like pandas describe method
def univariate_stats(df):
    #df.drop('id', axis=1, inplace=True)
    output_df = pd.DataFrame(columns=['Count', 'Missing', 'Unique', 'Dtype', 'IsNumeric', 'Mode', 'Mean', 'Min', '25%', 'Median', '75%', 'Max', 'Std', 'Skew', 'Kurt'])
    for col in df:
        if is_numeric_dtype(df[col]):
            output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype, is_numeric_dtype(df[col]), df[col].mode().values[0], df[col].mean(), df[col].min(), df[col].quantile(.25), df[col].median(), df[col].quantile(.75), df[col].max(), df[col].std(), df[col].skew(), df[col].kurt() ]
        else:
            output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype, is_numeric_dtype(df[col]), df[col].mode().values[0], '-', '-', '-', '-', '-', '-', '-', '-', '-' ]
    return output_df.sort_values(by=['IsNumeric', 'Unique'], ascending=False)


# this just an intermediate function that will be used in bivstats for one-way ANOVA
def anova(df, feature, label):
    groups = df[feature].unique()
    df_grouped = df.groupby(feature)
    group_labels = []
    for g in groups:
        g_list = df_grouped.get_group(g)
        group_labels.append(g_list[label])

    return stats.f_oneway(*group_labels)


def plot_histograms(df_train, df_test, target_col, n_cols=3):
    n_rows = (len(df_train.columns) - 1) // n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()

    for i, var_name in enumerate(df_train.columns.tolist()):
        #print(var_name)
        ax = axes[i]
        sns.distplot(df_train[var_name], kde=True, ax=ax, label='Train')      # plot train data

        if var_name != target_col:
            sns.distplot(df_test[var_name], kde=True, ax=ax, label='Submission')    # plot test data
        ax.set_title(f'{var_name} Distribution')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_boxplot(df, title='', drop_cols=[config.TARGET_COL], n_cols=3):
    sns.set_style('darkgrid')
    cols = df.columns #.drop(drop_cols)
    n_rows = (len(cols) - 1) // n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 4*n_rows))

    # reset index to avoid duplicate index issues
    df = df.reset_index(drop=True)

    for i, var_name in enumerate(cols):
        #print(var_name)
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        sns.boxplot(data=df, y=var_name, ax=ax, showmeans=True,
                    meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", "markersize":"5"})
        ax.set_title(f'{var_name}')

    fig.suptitle(f'{title} boxplot', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_counts(df, cols):
    _, ax = plt.subplots(nrows=3, ncols=2, figsize=(32, 36))

    for index, col in enumerate(cols):
        r = index // 2
        c = index % 2
        g = sns.countplot(data=df, x=col, ax=ax[r][c], width=0.6)
        g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right", fontsize=18)
        ax[r][c].set_title(f'{col} distribution', fontsize=24)
        plt.tight_layout()


# function to calculate bivariate stats; Pearson' correlation, p-value and one-way ANOVA
def bivstats(df, label=config.TARGET_COL):
    # Create an empty DataFrame to store output
    output_df = pd.DataFrame(columns=['Stat', '+/-', 'Effect size', 'p-value'])

    for col in df:
        if col != label:
            if df[col].isnull().sum() == 0:
                if is_numeric_dtype(df[col]):   # Only calculate r,
                    r, p = stats.pearsonr(df[label], df[col])
                    output_df.loc[col] = ['r', np.sign(r), abs(round(r, 3)), round(p,6)]
                else:
                    F, p = anova(df[[col, label]], col, label)
                    output_df.loc[col] = ['F', '', round(F, 3), round(p,6)]
            else:
                output_df.loc[col] = [np.nan, np.nan, np.nan, np.nan]

    return output_df.sort_values(by=['Effect size', 'Stat'], ascending=[False, False])


def plot_corr_matrix(df, cols):
    corr_matrix = df[cols + [config.TARGET_COL]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype = bool))

    cmap = sns.diverging_palette(100, 7, s = 75, l = 40, n = 20, center = 'light', as_cmap = True)

    fig, ax = plt.figure(figsize = (25, 10))
    sns.heatmap(corr_matrix, annot = True, cmap = cmap, fmt = '.2f', center = 0,
                annot_kws = {'size': 12}, ax = ax, mask = mask).set_title('Correlations Among Features ({df} set)')


def plot_scatter_with_fixed_col(df, fixed_col, hue=False, drop_cols=[], size=5, title=''):
    sns.set_style('darkgrid')
    if hue:
        cols = df.columns.drop([hue, fixed_col] + drop_cols, errors='ignore')
    else:
        cols = df.columns.drop([fixed_col] + drop_cols, errors='ignore')

    n_cols = 4
    n_rows = (len(cols) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size, size/n_cols*n_rows), sharex=False, sharey=False)
    fig.suptitle(f'training data scatter plot vs. target', fontsize=15, fontweight='bold', y=1)

    # reset index to avoid duplicate index issues
    df = df.reset_index(drop=True)

    for i, col in enumerate(cols):
        n_row = i // n_cols
        n_col = i % n_cols
        ax = axes[n_row, n_col]
        ax.set_xlabel(f'{col}', fontsize=6)
        ax.set_ylabel(f'{fixed_col}', fontsize=6)

        # Plot the scatterplot
        if hue:
            sns.scatterplot(data=df, x=col, y=fixed_col, hue=hue, ax=ax,
                            s=40, edgecolor='gray', alpha=0.3, palette='bright')
            ax.legend(title=hue, title_fontsize=12, fontsize=12)
        else:
            sns.scatterplot(data=df, x=col, y=fixed_col, ax=ax,
                            #s=2, edgecolor='gray', alpha=0.1
                            )

        ax.tick_params(axis='both', which='major', labelsize=2)
        ax.set_title(f'{col}', fontsize=8)

    plt.tight_layout()
    plt.show()