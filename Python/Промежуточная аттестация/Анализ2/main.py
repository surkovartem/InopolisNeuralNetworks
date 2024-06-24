import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu


# Получение data frame с колонками числовых значений
def get_numerical_data_frame(df):
    numerical = df.select_dtypes(include='number').columns
    return df[numerical]


# Получение анализа фрейма числовых значений
def get_analysis_numeric_columns(df):
    result = ''
    columns_set = set(df.columns.tolist())
    for column in columns_set:
        result += f'\n{get_percentage_empty_cells(df, column)}\n'
        result += f'Максимальное значение в столбце {column}: {df[column].max()}\n'
        result += f'Минимальное значение в столбце {column}: {df[column].min()}\n'
        result += f'Среднее значение в столбце {column}: {df[column].mean()}\n'
        result += f'Медиана в столбце {column}: {df[column].median()}\n'
        result += f'Дисперсия в столбце {column}: {df[column].var()}\n'
    return result


# Получение data frame с колонками категориальных значений
def get_categorical_data_frame(df):
    categorical = df.select_dtypes(exclude='number').columns
    return df[categorical]


# Получение анализа фрейма категориальных значений
def get_analysis_categorical_columns(df):
    result = ''
    columns_set = set(df.columns.tolist())
    for column in columns_set:
        result += f'\n{get_percentage_empty_cells(df, column)}\n'
        result += f'Количество уникальных значений для столбца {column}: {df[column].nunique()}\n'
    return result


# Получение доли пропусков в стобце (доля пустых ячеек)
def get_percentage_empty_cells(df, column_name):
    empty_cells_count = df[column_name].isnull().sum()
    total_cells_count = len(df[column_name])
    empty_cells_ratio = empty_cells_count / total_cells_count
    return str(f"Доля пустых ячеек в столбце {column_name}: "
               f"{empty_cells_ratio * 100:.2f}%")


# Проверка гипотезы о том, что данные взяты из нормально распределённой совокупности
def checking_distribution_data(data_a, data_b):
    stat_shapiro_a, p_value_shapiro_a = shapiro(data_a)
    is_normal_a_method = p_value_shapiro_a > 0.05
    if is_normal_a_method:
        print("Данные выборки 'A' могут соответствовать нормальному распределению")
    else:
        print("Данные выборки 'A' не соответствуют нормальному распределению")

    stat_shapiro_b, p_value_shapiro_b = shapiro(data_b)
    is_normal_b_method = p_value_shapiro_b > 0.05
    if is_normal_b_method:
        print("Данные выборки 'B' могут соответствовать нормальному распределению")
    else:
        print("Данные выборки 'B' не соответствуют нормальному распределению")

    return [is_normal_a_method, is_normal_b_method]


# t-критерий Стьюдента
def test_of_student(data_a, data_b):
    t_stat, p_value = ttest_ind(data_a, data_b)
    print("t-статистика:", t_stat)
    print("p-значение:", p_value)

    if p_value < 0.05:
        print("Различия между выборками статистически значимы.")
    else:
        print("Различия между выборками статистически не значимы.")

    if t_stat < 0:
        print("Уровень счастья, в рамках ВВП на душу населения обратно пропорционально.")
    else:
        print("Уровень счастья, в рамках ВВП на душу населения прямо пропорционально.")


# u-критерий Манна-Уитни
def test_mann_whitney(data_a, data_b):
    stat, p = mannwhitneyu(data_a, data_b)

    print("Статистика:", stat)
    print("p-значение:", p)

    if p < 0.05:
        print("Различия между выборками статистически значимы.")
    else:
        print("Различия между выборками статистически не значимы.")

    if stat < 0:
        print("Уровень счастья, в рамках ВВП на душу населения обратно пропорционально.")
    else:
        print("Уровень счастья, в рамках ВВП на душу населения прямо пропорционально.")


df = pd.read_csv('2015.csv')
row_count, columns_count = df.shape
print(f'Количество строк: {row_count}\n'
      f'Колимчество столбцов: {columns_count}')

# Дата фрейм, содердащий в себе, только числовые колонки
numerical_df = get_numerical_data_frame(df)
# Дата фрейм, содердащий в себе, только категориальные колонки
categorical_df = get_categorical_data_frame(df)

# Разведочный анализ для числовых переменных
analysis_numerical_df = get_analysis_numeric_columns(numerical_df)
print(analysis_numerical_df)
# Разведочный анализ для категориальных переменных
analysis_categorical_df = get_analysis_categorical_columns(categorical_df)
print(analysis_categorical_df)

plt.figure(figsize=(50, 10))
sns.barplot(x=df['Region'], y=df['Health (Life Expectancy)'])
plt.show()

plt.figure(figsize=(50, 10))
sns.barplot(x=df['Region'], y=df['Economy (GDP per Capita)'])
plt.show()

# Экономические показатели страны Сингапур
economy_southeastern_asia = df[df['Region'] == 'Southeastern Asia']['Economy (GDP per Capita)'].tolist()
# Экономические показатели страны Катар
economy_sub_saharan_africa = df[df['Region'] == 'Sub-Saharan Africa']['Economy (GDP per Capita)'].tolist()

distribution_data = checking_distribution_data(economy_southeastern_asia, economy_sub_saharan_africa)
distribution_southeastern_asia = distribution_data[0]
distribution_sub_saharan_africa = distribution_data[1]

if (distribution_southeastern_asia == True) and (distribution_sub_saharan_africa == True):
    print("\nОбе выборки нормально распределенны. Использование t-критерия Стьюдента...")
    test_of_student(distribution_southeastern_asia, distribution_sub_saharan_africa)
else:
    print("\nОбе выборки нормально не распределены. Использование u-критерий Манна-Уитни... ")
    test_mann_whitney(distribution_southeastern_asia, distribution_sub_saharan_africa)
