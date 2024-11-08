import warnings
from .constants import MAD_CONFORM


def _inform_(df, high_Z, conf):
    """Выбирает и сортирует по выбранным Z_stats, информируя или нет.
    """

    if isinstance(high_Z, int):
        if conf is not None:
            dd = df[['Теор', 'Факт', 'Z_статистика'
                     ]].sort_values('Z_статистика', ascending=False).head(high_Z)
            print(f'\nЭлементы с наибольшими {high_Z} Z-значениями:\n')
        # Тест суммирования
        else:
            dd = df[['Expected', 'Факт', 'Абс_Разность'
                     ]].sort_values('Абс_Разность', ascending=False
                                    ).head(high_Z)
            print(f'\nЭлементы с наибольшими {high_Z} абсолютными отклонениями:\n')
    else:
        if high_Z == 'pos':
            m1 = df.Dif > 0
            m2 = df.Z_статистика > conf
            dd = df[['Теор', 'Факт', 'Z_статистика'
                     ]].loc[m1 & m2].sort_values('Z_статистика', ascending=False)
            print('\nЗначения со значимыми положительными отклонениями:\n')
        elif high_Z == 'neg':
            m1 = df.Dif < 0
            m2 = df.Z_статистика > conf
            dd = df[['Теор', 'Факт', 'Z_статистика'
                     ]].loc[m1 & m2].sort_values('Z_статистика', ascending=False)
            print('\nЗначения со значимыми отрицательными отклонениями:\n')
        else:
            dd = df[['Теор', 'Факт', 'Z_статистика'
                     ]].loc[df.Z_статистика > conf].sort_values('Z_статистика',
                                                           ascending=False)
            print('\nЗначения со значимыми отклонениями:\n')
    print(dd)


def _report_mad_(digs, MAD):
    """Отчет о среднем абсолютном отклонении теста и сравнение его с критическими значениями
    """
    print(f'Среднее абсолютное отклонение (MAD): {MAD:.6f}')
    if digs != -2:
        mads = MAD_CONFORM[digs]
        if MAD <= mads[0]:
            print(f'MAD <= {mads[0]:.6f}: Тесное соответствие.\n')
        elif MAD <= mads[1]:
            print(f'{mads[0]:.6f} < MAD <= {mads[1]:.6f}: '
                  'Допустимое соответствие.\n')
        elif MAD <= mads[2]:
            print(f'{mads[1]:.6f} < MAD <= {mads[2]:.6f}: '
                  'С погрешностью приемлемое соответствие.\n')
        else:
            print(f'MAD > {mads[2]:.6f}: Несоответствие.\n')
    else:
        print("Проверка соответствия MAD для этого теста не предусмотрена.\n")


def _report_KS_(KS, crit_KS):
    """Отчет о статистике теста Колмогорова-Смирнова и сравнение ее с критическими
    значениями, в зависимости от уровня доверия
    """
    result = 'Cоответствие' if KS <= crit_KS else 'Несоответствие'
    print(f"\n\tСтатистика Колмогорова-Смирнова: {KS:.6f}",
          f"\n\tКритическое значение: {crit_KS:.6f} -- {result}")


def _report_chi2_(chi2, CRIT_CHI2):
    """Отчет о статистике теста хи-квадрат и сравнение ее с критическими значениями,
    в зависимости от уровня доверия
    """
    result = 'Cоответствие' if chi2 <= CRIT_CHI2 else 'Несоответствие'
    print(f"\n\tСтатистика Хи-квадрат: {chi2:.6f}",
          f"\n\tКритическое значение: {CRIT_CHI2:.6f} -- {result}")


def _report_Z_(df, high_Z, crit_Z):
    """Отчет о Z-значениях теста и сравнение их с критическим значением,
    в зависимости от уровня доверия
    """
    print(f"\n\tКритическое значение Z-статистики:{crit_Z}.")
    _inform_(df, high_Z, crit_Z)


def _report_summ_(test, high_diff):
    """Отчет о тесте суммирования абсолютных разностей между найденными и
    ожидаемыми долями

    """
    if high_diff is not None:
        print(f'\nНаибольшие {high_diff} абсолютные разности:\n')
        print(test.sort_values('Абс_Разность', ascending=False).head(high_diff))
    else:
        print('\nНаибольшие абсолютные разности:\n')
        print(test.sort_values('Абс_Разность', ascending=False))


def _report_bhattac_coeff_(bhattac_coeff):
    """
    """
    print(f"Коэффициент Бхаттачарьи: {bhattac_coeff:6f}\n")


def _report_bhattac_dist_(bhattac_dist):
    """
    """
    print(f"Расстояние Бхаттачарьи: {bhattac_dist:6f}\n")


def _report_kl_diverg_(kl_diverg):
    """
    """
    print(f"Расстояние Кульбака-Лейблера: {kl_diverg:6f}\n")


def _report_test_(test, high=None, crit_vals=None):
    """Основная отчетная функция. Получает аргументы для отчета, инициирует
    процесс и вызывает нужную отчетную вспомогательную функцию(ии), в зависимости
    от Теста.
    """
    print('\n', f'  {test.name}  '.center(50, '#'), '\n')
    if not 'Summation' in test.name:
        _report_mad_(test.digs, test.MAD)
        _report_bhattac_coeff_(test.bhattacharyya_coefficient)
        _report_bhattac_dist_(test.bhattacharyya_distance)
        _report_kl_diverg_(test.kullback_leibler_divergence)
        if test.confidence is not None:
            print(f"На доверительном интервале {test.confidence}%: ")
            _report_KS_(test.KS, crit_vals['KS'])
            _report_chi2_(test.chi_square, crit_vals['chi2'])
            _report_Z_(test, high, crit_vals['Z'])
        else:
            print('Уровень доверия в настоящее время `None`. Задайте уровень доверия, '
                  'чтобы сгенерировать сравнимые критические значения.')
            if isinstance(high, int):
                _inform_(test, high, None)
    else:
        _report_summ_(test, high)


def _report_mantissa_(stats, confidence):
    """Выводит статистику мантисс и их соответствующие справочные значения

    Args:
        stats (dict):
    """
    print("\n", '  Тест мантисс  '.center(52, '#'))
    print(f"\nСреднее мантиисс: {stats['Mean']:.6f}."
          "\tТеор.: 0.5")
    print(f"Отклонение мантисс: {stats['Var']:.6f}."
          "\tТеор.: 0.08333")
    print(f"Асимметрия мантисс: {stats['Skew']:.6f}."
          "\tТеор.: 0.0")
    print(f"Эксцесс мантисс: {stats['Kurt']:.6f}."
          "\tТеор.: -1.2")
    print("\nСтатистика Колмогорова-Смирнова для распределения Мантисс: "
          f"{stats['KS']:.6f}.\nКритическое значение "
          f"на доверительном интервале {confidence}%: {stats['KS_critical']:.6f} -- "
          f"{'Соответствие' if stats['KS'] < stats['KS_critical'] else 'Несоответствие'}\n")


def _deprecate_inform_(verbose, inform):
    """
    Рaises:
        FutureWarning: если используется аргумент `inform` (будет удален в будущих версиях).
    """
    if inform is None:
        return verbose
    else:
        warnings.warn('Параметр `inform` будет удален в будущих версиях. Используйте `verbose` вместо него.',
                      FutureWarning)
        return inform
