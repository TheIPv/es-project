from pandas import Series, DataFrame
from numpy import array, arange, log10, ndarray
from .expected import _get_expected_digits_
from .constants import DIGS, REV_DIGS
from .stats import Z_статистика
from .checks import _check_num_array_, _check_sign_, _check_decimals_

def _set_N_(len_df, limit_N):
    """"""
    # Присваивает N предельное значение или длину серии
    if limit_N is None or limit_N > len_df:
        return max(1, len_df)
    # Проверка на положительное целое значение limit_N
    else:
        if limit_N < 0 or not isinstance(limit_N, int):
            raise ValueError("limit_N должен быть None или положительным целым числом.")
        else:
            return max(1, limit_N)

def get_mantissas(arr):
    """Вычисляет мантиссы, нецелую часть логарифма числа.

    Args:
        arr: массив целых или вещественных чисел

    Returns:
        Массив вещественных чисел с мантиссами логарифмов
    """
    log_a = abs(log10(arr))
    return log_a - log_a.astype(int)  # число - его целая часть

def input_data(given):
    """Обрабатывает и преобразует входные данные

    Args:
        given: ndarray, Series или кортеж с DataFrame и именем столбца для анализа

    Returns:
        Оригинальные входные данные и результат их первой предварительной обработки, если требуется.
    """
    if type(given) == Series:
        data = chosen = given
    elif type(given) == ndarray:
        data = given
        chosen = Series(given)
    elif type(given) == tuple:
        if (type(given[0]) != DataFrame) | (type(given[1]) != str):
            raise TypeError('Кортеж данных должен состоять из DataFrame pandas и имени (str) выбранного столбца, в этом порядке')
        data = given[0]
        chosen = given[0][given[1]]
    else:
        raise TypeError("Неверный тип входных данных. Проверьте документацию.")
    return data, chosen

def set_sign(data, sign="all"):
    """"""
    sign = _check_sign_(sign)

    if sign == 'all':
        data.seq = data.seq.loc[data.seq != 0]
    elif sign == 'pos':
        data.seq = data.seq.loc[data.seq > 0]
    else:
        data.seq = data.seq.loc[data.seq < 0]

    return data.dropna()

def get_times_10_power(data, decimals=2):
    """"""
    decimals = _check_decimals_(decimals)

    ab = data.seq.abs()

    if data.seq.dtype == 'int':
        data['ZN'] = ab
    else:
        if decimals == 'infer':
            data['ZN'] = ab.astype(str).str\
                .replace('.', '', regex=False)\
                .str.lstrip('0')\
                .str[:5].astype(int)
        else:
            data['ZN'] = (ab * (10 ** decimals)).astype(int)
    return data

def get_digs(data, decimals=2, sign="all"):
    """  """
    df = DataFrame({'seq': _check_num_array_(data)})

    df = set_sign(df, sign=sign)

    df = get_times_10_power(df, decimals=decimals)

    # Первые цифры
    for col in ['F1D', 'F2D', 'F3D']:
        temp = df.ZN.loc[df.ZN >= 10 ** (REV_DIGS[col] - 1)]
        df[col] = (temp // 10 ** ((log10(temp).astype(int)) -
                                  (REV_DIGS[col] - 1)))
        # заполняем NAN значениями -1, которые не используются для цифр,
        # чтобы позже их отбросить.
        df[col] = df[col].fillna(-1).astype(int)
    # Вторая цифра
    temp_sd = df.loc[df.ZN >= 10]
    df['SD'] = (temp_sd.ZN // 10**((log10(temp_sd.ZN)).astype(int) -
                                   1)) % 10
    df['SD'] = df['SD'].fillna(-1).astype(int)
    # Последние две цифры
    temp_l2d = df.loc[df.ZN >= 1000]
    df['L2D'] = temp_l2d.ZN % 100
    df['L2D'] = df['L2D'].fillna(-1).astype(int)
    return df

def get_found_proportions(data):
    """"""
    counts = data.value_counts()
    # получаем их относительные частоты
    proportions = data.value_counts(normalize=True)
    # создаем DataFrame из них
    return DataFrame({'Counts': counts, 'Факт': proportions}).sort_index()

def join_expect_found_diff(data, digs):
    """"""
    dd =_get_expected_digits_(digs).join(data).fillna(0)
    # создаем колонку с абсолютными разницами
    dd['Dif'] = dd.Факт - dd.Теор
    dd['Абс_Разность'] = dd.Dif.abs()
    return dd

def prepare(data, digs, limit_N=None, simple=False):
    """Преобразует исходную последовательность чисел в DataFrame, уменьшенный
    по вхождениям выбранных цифр, создает другие вычисленные колонки
    """
    df = get_found_proportions(data)
    dd = join_expect_found_diff(df, digs)
    if simple:
        del dd['Dif']
        return dd
    else:
        N = _set_N_(len(data), limit_N=limit_N)
        dd['Z_статистика'] = Z_статистика(dd, N)
        return N, dd

def subtract_sorted(data):
    """Вычитает из отсортированной последовательности элементы друг из друга, отбрасывая нули.
    Используется во Вторичном Тесте
    """
    temp = data.copy().sort_values(ignore_index=True)
    temp = (temp - temp.shift(1)).dropna()
    return temp.loc[temp != 0]

def prep_to_roll(start, test):
    """Используется в rolling mad и rolling mean, готовит каждый тест и
    их ожидаемые пропорции для последующего применения к подмножеству Series
    """
    if test in [1, 2, 3]:
        start[DIGS[test]] = start.ZN // 10 ** ((
            log10(start.ZN).astype(int)) - (test - 1))
        start = start.loc[start.ZN >= 10 ** (test - 1)]

        ind = arange(10 ** (test - 1), 10 ** test)
        Exp = log10(1 + (1. / ind))

    elif test == 22:
        start[DIGS[test]] = (start.ZN // 10 ** ((
            log10(start.ZN)).astype(int) - 1)) % 10
        start = start.loc[start.ZN >= 10]

        Expec = log10(1 + (1. / arange(10, 100)))
        temp = DataFrame({'Теор': Expec, 'Вторые цифры':
                          array(list(range(10)) * 9)})
        Exp = temp.groupby('Вторые цифры').sum().values.reshape(10,)
        ind = arange(0, 10)

    else:
        start[DIGS[test]] = start.ZN % 100
        start = start.loc[start.ZN >= 1000]

        ind = arange(0, 100)
        Exp = array([1 / 99.] * 100)

    return Exp, ind

def mad_to_roll(arr, Exp, ind):
    """Среднее абсолютное отклонение(MAD), используемое в rolling-функции
    """
    prop = arr.value_counts(normalize=True).sort_index()

    if len(prop) < len(Exp):
        prop = prop.reindex(ind).fillna(0)

    return abs(prop - Exp).mean()

def mse_to_roll(arr, Exp, ind):
    """Среднеквадратичное отклонение, используемое в rolling-функции
    """
    temp = arr.value_counts(normalize=True).sort_index()

    if len(temp) < len(Exp):
        temp = temp.reindex(ind).fillna(0)

    return ((temp - Exp) ** 2).mean()
