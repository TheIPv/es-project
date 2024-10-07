from pandas import Series
from numpy import array, ndarray
from .constants import DIGS, REV_DIGS, CONFS

def _check_digs_(digs):
    """Проверяет возможные значения для параметра digs в тестах Первых Цифр"""
    if digs not in [1, 2, 3]:
        raise ValueError("Значение, присвоенное параметру -digs-, было {digs}. Значение должно быть 1, 2 или 3.")

def _check_test_(test):
    """Проверяет выбранный тест, как для int, так и для str значений"""
    if isinstance(test, int):
        if test in DIGS.keys():
            return test
        else:
            raise ValueError(f'Тест был установлен на {test}. Должно быть одно из {DIGS.keys()}')
    elif isinstance(test, str):
        if test in REV_DIGS.keys():
            return REV_DIGS[test]
        else:
            raise ValueError(f'Тест был установлен на {test}. Должно быть одно из {REV_DIGS.keys()}')
    else:
        raise ValueError('Неправильное значение выбрано для параметра теста. Возможные значения:\n {list(DIGS.keys())} для int и\n {list(REV_DIGS.keys())} для строк.')

def _check_decimals_(decimals):
    """"""
    if isinstance(decimals, int):
        if (decimals < 0):
            raise ValueError("Параметр -decimals- должен быть int >= 0, или 'infer'.")
    else:
        if decimals != 'infer':
            raise ValueError("Параметр -decimals- должен быть int >= 0, или 'infer'.")
    return decimals

def _check_sign_(sign):
    """"""
    if sign not in ['all', 'pos', 'neg']:
        raise ValueError("Параметр -sign- должен быть одним из следующих: " "'all', 'pos' или 'neg'.")
    return sign

def _check_confidence_(confidence):
    """"""
    if confidence not in CONFS.keys():
        raise ValueError("Значение параметра -confidence- должно быть одним из следующих:\n {list(CONFS.keys())}")
    return confidence

def _check_high_Z_(high_Z):
    """"""
    if not high_Z in ['pos', 'all']:
        if not isinstance(high_Z, int):
            raise ValueError("Параметр -high_Z- должен быть 'pos', 'all' или int.")
    return high_Z

def _check_num_array_(data):
    """"""
    if (not isinstance(data, ndarray)) & (not isinstance(data, Series)):
        print('\n`data` не является numpy NDarray или pandas Series.' ' Попытка преобразовать...')
        try:
            data = array(data)
        except:
            raise ValueError('Не удалось преобразовать данные. Проверьте ввод.')
        print('\nПреобразование успешно.')

        try:
            data = data.astype(float)
        except:
            raise ValueError('Не удалось преобразовать данные. Проверьте ввод.')
    else:
        if data.dtype not in [int, float]:
            try:
                data = data.astype(float)
            except:
                raise ValueError('Не удалось преобразовать данные. Проверьте ввод.')
    return data
