# Импорт библиотек

import pandas as pd
import numpy as np
from numpy import abs
import matplotlib.pyplot as plt

df = pd.read_excel('ЖО Дубнадорстрой.xlsx',converters={'Счет Кт': str, 'Счет Дт': str})
OSV = pd.read_excel('ОСВ Дубнадорстрой.xlsx').round(decimals=2).set_index('Счет')

df.head(5)


# Тест целостности данных

names = '000, 001, 002, 003, 003.01, 003.02, 004, 004.1, 004.02, 004.К, 005, 006, 007, 008, 008.1, 008.21, 009, 009.01, 009.21, 010, 011, 012, 012.01, 012.02, ГТД, КВ, МЦ, МЦ.02, МЦ.03, МЦ.04, НЕ, НЕ.01, НЕ.01.1, НЕ.01.9, НЕ.02, НЕ.02.1, НЕ.02.9, НЕ.03, НЕ.04, ОТ, ОТ.01, ОТ.02, ОТ.03, РВ, РВ.1, РВ.2, РВ.3, РВ.4, УСН, УСН.01, УСН.01, УСН.02, УСН.03, УСН.04, УСН.21, УСН.22, УСН.23, УСН.24, Я75, Я81.01, Я81, Я80, Я80.02, Я80.01, Я80.09, Я75.01, Я81.09, Я81.02, Я75.02, Я69.06.5, Я01.К, Я96, Я96.01'
namesofaccs = [accs.strip() for accs in names.split(',')]
result = df[df['СчетДт'].isnull() | df['СчетКт'].isnull()]
result = result.loc[~result['СчетКт'].isin(namesofaccs) & ~result['СчетДт'].isin(namesofaccs)]

if (result.shape[0] == 0):
    print("Проверка целостности данных успешно завершена")
else:
    print("Проверка целостности данных выполнена с ошибками")

# Тест математической правильности

resultosv = OSV.loc[(OSV['Сальдо начальное Дт'].fillna(0) - OSV['Сальдо начальное Кт'].fillna(0) + OSV['Обороты Дт'].fillna(0) - OSV['Обороты Кт'].fillna(0) - OSV['Сальдо конечное Дт'].fillna(0) + OSV['Сальдо конечное Кт'].fillna(0)).abs() > 0.9]

if (resultosv.shape[0] == 0):
    print("Проверка математической правильности успешно завершена")
else:
    print("Проверка математической правильности выполнена с ошибками")

# Тест полноты выгрузки

journal_sum = df[['СчетДт', 'СчетКт', 'Сумма']]
journal_sum = journal_sum.fillna(0)
journal_sum_dt = journal_sum.groupby(['СчетДт'], as_index=False)['Сумма'].sum()
journal_sum_dt['Количество Счет Дт'] = journal_sum.groupby(['СчетДт'])['СчетДт'].count()
journal_sum_kt = journal_sum.groupby(['СчетКт'], as_index=False)['Сумма'].sum()
journal_sum_kt['Количество Счет Кт'] = journal_sum.groupby(['СчетКт'])['СчетКт'].count()
journal_sum_kt = journal_sum_kt.rename(columns = {'Сумма':'Обороты Кт по выгрузке'})
journal_sum_dt = journal_sum_dt.rename(columns = {'Сумма':'Обороты Дт по выгрузке'})
journal_sum_dt = journal_sum_dt.set_index('СчетДт')
journal_sum_kt = journal_sum_kt.set_index('СчетКт')
osvjournal = pd.concat([OSV, journal_sum_kt.reindex(OSV.index)], axis=1)
osvjournal = pd.concat([osvjournal, journal_sum_dt.reindex(OSV.index)], axis=1)
osvjournal['Обороты Дт'] = osvjournal['Обороты Дт'].fillna(0)
osvjournal['Обороты Кт'] = osvjournal['Обороты Кт'].fillna(0)
osvjournal['Обороты Дт по выгрузке'] = osvjournal['Обороты Дт по выгрузке'].fillna(0)
osvjournal['Обороты Кт по выгрузке'] = osvjournal['Обороты Кт по выгрузке'].fillna(0)
osvjournal['Обороты Дт разница'] = osvjournal['Обороты Дт'].round(2) - osvjournal['Обороты Дт по выгрузке'].round(2)
osvjournal['Обороты Кт разница'] = osvjournal['Обороты Кт'].round(2) - osvjournal['Обороты Кт по выгрузке'].round(2)
osvresult = osvjournal.loc[(osvjournal['Обороты Дт разница'] != 0) | (osvjournal['Обороты Кт разница'] != 0)]
osvresult = osvresult.rename(columns={'Обороты Дт': 'Обороты Дт по ОСВ', 'Обороты Кт': 'Обороты Кт по ОСВ'})
result = osvresult[["Обороты Дт по ОСВ","Обороты Кт по ОСВ", "Обороты Дт по выгрузке", "Обороты Кт по выгрузке", "Обороты Дт разница", "Обороты Кт разница"]]
result = result.reset_index()

result

if (len(osvresult) != 0):
    print("Проверка полноты выгрузки выполнена с ошибками")
else:
    print("Проверка полноты выгрузки успешно завершена")
    result

"""# **Проверка законом Бенфрода**

## **Вспомогательный код**
"""

import ru_benford as bf
benf = bf.Benford(df['Сумма'], decimals=2, confidence=99)

benf.summation()

# Добавляем признак: является ли сумма сторнированной (1) или нет (0)
df["Сторно"] = df["Сумма"].apply(lambda x: 0 if x > 0 else 1)

# Убираем строки с пустыми значениями поля Сумма
df = df[df['Сумма'].notnull()]

# Для отрицательных сумм меняем знак
df.loc[df["Сумма"] < 0, "Сумма"] = -df["Сумма"]

# Оставляем значения, больше 10, чтобы на них можно было провести все тесты
df = df[df['Сумма'] > 10]

# Меняем индексы
df = df.reset_index(drop=True)

"""## **2.1 Базовые тесты**

### **2.1.1 Тест первой цифры**
"""

benf.F1D.report()

"""### **2.1.2 Тест второй цифры**"""

benf.SD.report()

"""### **2.1.3 Тест первых двух цифр**"""

benf.F2D.report()

"""## **2.2 Продвинутые тесты**

### **2.2.1 Тест суммирования**
"""

sm = bf.summation(df['Сумма'], decimals=2)

"""### **2.2.2 Тест второго порядка**"""

benf.sec_order()
benf.F2D_sec.report()

"""### **2.2.3 Тест мантисс**"""

mant = bf.mantissas(df['Сумма'])

"""## **2.3 Связанные тесты**

### **2.3.1 Тест дублирования сумм**
"""

#Считаем частоту появления сумм

temp = df.groupby('Сумма')['Сумма'].count()/len(df)
sum_ = list(temp.index)
frequency = list(temp)
df_temp = pd.DataFrame({"Сумма": sum_, "Частота суммы": frequency})

df = df.merge(df_temp, on = 'Сумма')

most_frequent_values = df['Сумма'].value_counts().head(10)

sorted_counts = most_frequent_values.sort_values(ascending=False)

bar_width = 0.5  # Увеличиваем ширину столбцов
bar_positions = [i + bar_width / 2 for i in range(len(sorted_counts))]  # Смещаем позиции столбцов

plt.figure(figsize=(10, 6))  # Увеличиваем размер графика
plt.bar(bar_positions, sorted_counts.values, width=bar_width, align='center', color=(0/255, 121/255, 140/255))  # Устанавливаем цвет в RGB
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Тест дублирования сумм')
plt.xticks(bar_positions, sorted_counts.index, rotation=45)  # Вращаем подписи для лучшей читаемости
plt.tight_layout()  # Оптимально размещаем элементы на графике
plt.show()

"""### **2.3.2 Тест двух последних цифр**"""

benf.L2D.report()

"""### **2.3.3 Оценка коэффициента искажения**"""

result = df.loc[df['Сумма'] >= 10]
result.loc[result['Сумма'] != 0, ['Сумма']] = (10*result['Сумма']/(10**(np.log10(result['Сумма']).fillna(0)).astype(int)))

Avg_Found = result['Сумма'].sum()/result['Сумма'].count()
Avg_Expected = 90/(result['Сумма'].count()*(10**(1/result['Сумма'].count())-1))
Distortion = (Avg_Found-Avg_Expected)/Avg_Expected*100
Std = result['Сумма'].std()
Z_stat = Distortion/Std

print(f'Среднее Факт: {round(Avg_Found, 2)}')
print(f'Среднее Теор: {round(Avg_Expected, 2)}')
print(f'Коэффициент искажения: {round(Distortion, 2)}%')
print(f'Z-статистика: {round(Z_stat, 2)}')
print('Критическое значение: 2,57')

if (Distortion < 0):
    print('Вывод: Значения в массиве занижены')
else:
    print('Вывод: Значения в массиве завышены')
if (abs(Z_stat) > 2,57):
    print('Коэффициент искажения является существенным')
else:
    print('Коэффициент искажения является несущественным')

"""# **Оценка риска**

# TBD

# **Экспорт итогов**

## **2. Закон Бенфорда**
"""

benf.summation()

initial_size = len(benf.chosen)
tested_sample_size = len(benf.base)
data = (
    f"'F1D': {benf._discarded['F1D']}\n; "
    f"'F2D': {benf._discarded['F2D']}\n; "
    f"'F3D': {benf._discarded['F3D']}\n; "
    f"'SD': {benf._discarded['SD']}\n; "
    f"'L2D': {benf._discarded['L2D']};"
)

# Путь для сохранения Excel-файла
output_path = 'benford_test_report.xlsx'

# Запись данных в Excel
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    # Создание листа "Результаты тестов Бенфорда"
    worksheet = writer.book.add_worksheet('Результаты тестов')

    # Запись описания и значений
    worksheet.write('A1', 'Исходный размер выборки')
    worksheet.write('B1', initial_size)

    worksheet.write('A2', 'Количество значений, на которых проведен тест')
    worksheet.write('B2', tested_sample_size)

    worksheet.write('A3', 'Исключенные значения для каждого теста')
    worksheet.write('B3', data)

# Файл будет сохранен в вашем текущем каталоге
print(f'Отчет сохранен в файле: {output_path}')

"""## **3. Тест первой, второй и 1и2 цифры**"""



"""## **4. Тест суммирования**"""


"""## **5. Тест второго порядка**"""

import io
import sys
import pandas as pd
import re

# Перехватываем вывод команды benf.F2D_sec.report()
benf.sec_order()

# Создаем строковый буфер для перехвата вывода
report_output = io.StringIO()
sys.stdout = report_output

# Выполняем команду, которая выводит результаты
image_path = 'plot.png'
benf.F2D_sec.report(save_plot = image_path)

# Возвращаем стандартный вывод обратно
sys.stdout = sys.__stdout__

# Получаем содержимое вывода как строку
report_text = report_output.getvalue()

# Парсим данные с помощью регулярных выражений для создания таблицы
pattern = re.compile(r'(\d{1,2})\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)')
parsed_data = pattern.findall(report_text)

# Создаем списки для DataFrame с результатами
first_two_digits = []
expected = []
found = []
z_scores = []

for entry in parsed_data:
    first_two_digits.append(entry[0])
    expected.append(float(entry[1]))
    found.append(float(entry[2]))
    z_scores.append(float(entry[3]))

# Создаем DataFrame с четкими заголовками
df_results = pd.DataFrame({
    'Первые цифры': first_two_digits,
    'Теор': expected,
    'Факт': found,
    'Z_статистика': z_scores
})

# Удаляем строки таблицы из полного текста отчета и дублирующиеся заголовки
report_lines = [line.strip() for line in report_text.splitlines() if not pattern.match(line)]
report_lines_cleaned = []
skip_next = False

for line in report_lines:
    if "Теор" in line and "Факт" in line and "Z_статистика" in line:
        if not skip_next:
            skip_next = True
            continue
    if skip_next:
        skip_next = False
        continue
    report_lines_cleaned.append(line)

# Записываем данные в Excel
with pd.ExcelWriter('report.xlsx', engine='xlsxwriter') as writer:
    # Создаем лист для результатов анализа
    worksheet_results = writer.book.add_worksheet('тест второго порядка')

    # Записываем очищенный полный отчет, разбивая строки по пробелам
    for row_num, line in enumerate(report_lines_cleaned):
        columns = line.split()
        for col_num, cell_value in enumerate(columns):
            worksheet_results.write(row_num, col_num, cell_value)

    # Определяем начальную строку для вставки DataFrame
    start_row = len(report_lines_cleaned) + 2  # Добавляем небольшой отступ

    # Записываем DataFrame вниз под текстовым отчетом
    for col_num, header in enumerate(df_results.columns):
        worksheet_results.write(start_row, col_num, header)  # Записываем заголовки

    for row_num, (digit, exp, found_val, z) in enumerate(zip(df_results['Первые цифры'], df_results['Теор'], df_results['Факт'], df_results['Z_статистика']), start=start_row + 1):
        worksheet_results.write(row_num, 0, digit)  # First_2_Dig
        worksheet_results.write(row_num, 1, exp)  # Expected
        worksheet_results.write(row_num, 2, found_val)  # Found
        worksheet_results.write(row_num, 3, z)  # Z_score

    # Растягиваем ширину колонок
    for col_num, _ in enumerate(df_results.columns):
        max_width = max([len(str(value)) for value in df_results.iloc[:, col_num]] + [len(df_results.columns[col_num])])
        worksheet_results.set_column(col_num, col_num, max_width + 2)  # Добавляем небольшой отступ


    # Вставляем график в Excel со столбца K
    worksheet_results.insert_image('K1', image_path, {'x_scale': 0.5, 'y_scale': 0.5})

# Удаляем временный файл изображения
import os
os.remove(image_path)

"""## **6. Тест мантисс**"""

import io
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

# Выполняем команду, которая выводит результаты
image_path_1 = 'plot_1.png'
image_path_2 = 'plot_2.png'
report_output = io.StringIO()
sys.stdout = report_output

# Закрываем все открытые фигуры
plt.close('all')

# Генерируем графики и текст
mant = bf.mantissas(df['Сумма'])

# Возвращаем стандартный вывод обратно
sys.stdout = sys.__stdout__

# Получаем содержимое вывода как строку
report_text = report_output.getvalue()
report_lines = [line.strip() for line in report_text.splitlines()]

# Сохраняем все открытые графики по отдельности
figures = [plt.figure(i) for i in plt.get_fignums()]

# Сохраняем графики
figures[0].savefig(image_path_1)
figures[1].savefig(image_path_2)

# Закрываем все открытые фигуры
plt.close('all')

# Записываем данные в Excel
with pd.ExcelWriter('report_mantiss.xlsx', engine='xlsxwriter') as writer:
    # Создаем лист для результатов анализа
    worksheet_results = writer.book.add_worksheet('Тест Мантисс')

    # Записываем очищенный полный отчет, разбивая строки по пробелам
    for row_num, line in enumerate(report_lines):
        columns = line.split()
        for col_num, cell_value in enumerate(columns):
            worksheet_results.write(row_num, col_num, cell_value)

    # Вставляем графики в Excel
    worksheet_results.insert_image('A12', image_path_1, {'x_scale': 0.5, 'y_scale': 0.5})
    worksheet_results.insert_image('K1', image_path_2, {'x_scale': 0.5, 'y_scale': 0.5})

# Удаляем временные файлы изображений, если они существуют
os.remove(image_path_1)
os.remove(image_path_2)

"""## **7. Связанные тесты**"""

"# 7.1 Тест дублирования cумм"
temp = df.groupby('Сумма')['Сумма'].count()/len(df)
sum_ = list(temp.index)
frequency = list(temp)
df_temp = pd.DataFrame({"Сумма": sum_, "Частота суммы": frequency})

df = df.merge(df_temp, on = 'Сумма')

most_frequent_values = df['Сумма'].value_counts().head(10)

sorted_counts = most_frequent_values.sort_values(ascending=False)

bar_width = 0.5  # Увеличиваем ширину столбцов
bar_positions = [i + bar_width / 2 for i in range(len(sorted_counts))]  # Смещаем позиции столбцов

# Построение графика
fig, ax = plt.subplots(figsize=(10, 6))  # Используем fig и ax для сохранения фигуры
ax.bar(bar_positions, sorted_counts.values, width=bar_width, align='center', color=(0/255, 121/255, 140/255))
ax.set_xlabel('Значения')
ax.set_ylabel('Частота')
ax.set_title('Тест дублирования сумм')
ax.set_xticks(bar_positions)
ax.set_xticklabels(sorted_counts.index, rotation=45)

# Добавляем проверку данных и настраиваем границы осей
ax.set_xlim([min(bar_positions) - bar_width, max(bar_positions) + bar_width])
ax.set_ylim([0, max(sorted_counts.values) * 1.1])  # Увеличиваем верхнюю границу на 10% для видимости

plt.tight_layout()

# Сохраняем график
image_path = 'plot.png'
fig.savefig(image_path)  # Сохраняем график через fig.savefig
plt.close(fig)  # Закрываем конкретную фигуру

# Запись в Excel
with pd.ExcelWriter('report_connected_duplicate_amounts.xlsx', engine='xlsxwriter') as writer:
    worksheet_results = writer.book.add_worksheet('Тест дублирования сумм')
    worksheet_results.insert_image('A1', image_path)

# Удаляем временный файл
os.remove(image_path)

"# 7.2 Тест двух последних цифр"

import io
import sys
import pandas as pd
import re

# Перехватываем вывод команды benf.F2D_sec.report()
benf.sec_order()

# Создаем строковый буфер для перехвата вывода
report_output = io.StringIO()
sys.stdout = report_output

# Выполняем команду, которая выводит результаты
image_path = 'plot.png'
benf.L2D.report(save_plot = image_path)

# Возвращаем стандартный вывод обратно
sys.stdout = sys.__stdout__

# Получаем содержимое вывода как строку
report_text = report_output.getvalue()

# Парсим данные с помощью регулярных выражений для создания таблицы
pattern = re.compile(r'(\d{1,2})\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)')
parsed_data = pattern.findall(report_text)

# Создаем списки для DataFrame с результатами
first_two_digits = []
expected = []
found = []
z_scores = []

for entry in parsed_data:
    first_two_digits.append(entry[0])
    expected.append(float(entry[1]))
    found.append(float(entry[2]))
    z_scores.append(float(entry[3]))

# Создаем DataFrame с четкими заголовками
df_results = pd.DataFrame({
    'Первые цифры': first_two_digits,
    'Теор': expected,
    'Факт': found,
    'Z_статистика': z_scores
})

# Удаляем строки таблицы из полного текста отчета и дублирующиеся заголовки
report_lines = [line.strip() for line in report_text.splitlines() if not pattern.match(line)]
report_lines_cleaned = []
skip_next = False

for line in report_lines:
    if "Теор" in line and "Факт" in line and "Z_статистика" in line:
        if not skip_next:
            skip_next = True
            continue
    if skip_next:
        skip_next = False
        continue
    report_lines_cleaned.append(line)

# Записываем данные в Excel
with pd.ExcelWriter('report_connected_l2d.xlsx', engine='xlsxwriter') as writer:
    # Создаем лист для результатов анализа
    worksheet_results = writer.book.add_worksheet('тест двух последних цифр')

    # Записываем очищенный полный отчет, разбивая строки по пробелам
    for row_num, line in enumerate(report_lines_cleaned):
        columns = line.split()
        for col_num, cell_value in enumerate(columns):
            worksheet_results.write(row_num, col_num, cell_value)

    # Определяем начальную строку для вставки DataFrame
    start_row = len(report_lines_cleaned) + 2  # Добавляем небольшой отступ

    # Записываем DataFrame вниз под текстовым отчетом
    for col_num, header in enumerate(df_results.columns):
        worksheet_results.write(start_row, col_num, header)  # Записываем заголовки

    for row_num, (digit, exp, found_val, z) in enumerate(zip(df_results['Первые цифры'], df_results['Теор'], df_results['Факт'], df_results['Z_статистика']), start=start_row + 1):
        worksheet_results.write(row_num, 0, digit)  # First_2_Dig
        worksheet_results.write(row_num, 1, exp)  # Expected
        worksheet_results.write(row_num, 2, found_val)  # Found
        worksheet_results.write(row_num, 3, z)  # Z_score

    # Растягиваем ширину колонок
    for col_num, _ in enumerate(df_results.columns):
        max_width = max([len(str(value)) for value in df_results.iloc[:, col_num]] + [len(df_results.columns[col_num])])
        worksheet_results.set_column(col_num, col_num, max_width + 2)  # Добавляем небольшой отступ


    # Вставляем график в Excel со столбца K
    worksheet_results.insert_image('K1', image_path, {'x_scale': 0.5, 'y_scale': 0.5})

# Удаляем временный файл изображения
import os
os.remove(image_path)

"# 7.3 Оценка коэффициента искажения"

import pandas as pd
import numpy as np

# Расчет значений
result = df.loc[df['Сумма'] >= 10]
result.loc[result['Сумма'] != 0, ['Сумма']] = (10 * result['Сумма'] / (10 ** (np.log10(result['Сумма']).fillna(0)).astype(int)))

Avg_Found = result['Сумма'].sum() / result['Сумма'].count()
Avg_Expected = 90 / (result['Сумма'].count() * (10 ** (1 / result['Сумма'].count()) - 1))
Distortion = (Avg_Found - Avg_Expected) / Avg_Expected * 100
Std = result['Сумма'].std()
Z_stat = Distortion / Std

# Подготовка данных для записи в Excel
data = {
    'Показатель': [
        'Среднее Факт', 'Среднее Теор', 'Коэффициент искажения', 'Z-статистика', 'Критическое значение', 'Вывод по значениям', 'Значимость искажения'
    ],
    'Значение': [
        round(Avg_Found, 2), round(Avg_Expected, 2), f"{round(Distortion, 2)}%", round(Z_stat, 2), 2.57,
        'Значения в массиве занижены' if Distortion < 0 else 'Значения в массиве завышены',
        'Коэффициент искажения является существенным' if abs(Z_stat) > 2.57 else 'Коэффициент искажения является несущественным'
    ]
}

df_output = pd.DataFrame(data)

# Запись в файл Excel с настройками
with pd.ExcelWriter('report_connected_distortion.xlsx', engine='xlsxwriter') as writer:
    df_output.to_excel(writer, sheet_name='Анализ искажения', index=False, startrow=1)

    # Доступ к workbook и worksheet
    workbook = writer.book
    worksheet = writer.sheets['Анализ искажения']

    # Форматирование
    center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

    # Установка ширины столбцов по максимальной длине значений
    for i, col in enumerate(df_output.columns):
        max_len = max(df_output[col].astype(str).map(len).max(), len(col)) + 2  # Добавляем небольшой отступ
        worksheet.set_column(i, i, max_len, center_format)

    # Добавление заголовка
    worksheet.write(0, 0, 'Анализ искажения данных', workbook.add_format({'bold': True, 'align': 'center'}))
    worksheet.merge_range(0, 0, 0, len(df_output.columns) - 1, 'Анализ искажения данных', center_format)