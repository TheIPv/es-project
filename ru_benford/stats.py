from numpy import abs as nabs, errstate, linspace, log, sqrt, where
from .constants import CRIT_CHI2, CRIT_KS, MAD_CONFORM, DIGS

def Z_статистика(frame, N):
    """Расчет Z-статистики для изучаемых пропорций

    Args:
        frame: DataFrame с ожидаемыми пропорциями и уже рассчитанными
            абсолютными разницами между найденными и ожидаемыми пропорциями
        N: размер выборки

    Returns:
        Серия рассчитанных Z-значений
    """
    return (frame.Абс_Разность - (1 / (2 * N))) / sqrt(
           (frame.Теор * (1. - frame.Теор)) / N)

def chi_sq(frame, ddf, confidence, verbose=True):
    """Расчет статистики хи-квадрат найденных распределений и сравнение
    ее с критическим хи-квадратом такой выборки, в соответствии с
    выбранным уровнем доверия и степенями свободы - len(sample) -1.

    Args:
        frame: DataFrame с колонками Found, Expected и их разницей.
        ddf: Степени свободы для рассмотрения.
        confidence: Уровень доверия для поиска критического значения.
        verbose: печатает результат хи-квадрат и сравнивает его с критическим
            хи-квадратом для выборки. По умолчанию True.

    Returns:
        Рассчитанная статистика хи-квадрат и критическое хи-квадрат
            (в соответствии) с учетом степеней свободы и уровня доверия,
            для сравнения. None, если confidence равно None
    """
    if confidence is None:
        print('\nТест хи-квадрат требует уровня доверия, отличного от None.')
        return
    else:
        exp_counts = frame.Counts.sum() * frame.Теор
        dif_counts = frame.Counts - exp_counts
        found_chi = (dif_counts ** 2 / exp_counts).sum()
        crit_chi = CRIT_CHI2[ddf][confidence]
        if verbose:
            print(f"\nСтатистика хи-квадрат равна {found_chi:.4f}.\n"
                  f"Критическая статистика Хи-квадрат для этой выборки: {crit_chi}.")
        return (found_chi, crit_chi)

def chi_sq_2(frame):
    """Расчет статистики хи-квадрат найденных распределений

    Args:
        frame: DataFrame с колонками Found, Expected и их разницей.

    Returns:
        Рассчитанная статистика хи-квадрат
    """
    exp_counts = frame.Counts.sum() * frame.Теор
    dif_counts = frame.Counts - exp_counts
    return (dif_counts ** 2 / exp_counts).sum()

def kolmogorov_smirnov(frame, confidence, N, verbose=True):
    """Расчет теста Колмогорова-Смирнова для найденных распределений
    и сравнение его с критическим хи-квадратом такой выборки,
    в соответствии с выбранным уровнем доверия.

    Args:
        frame: DataFrame с распределениями Found и Expected.
        confidence: Уровень доверия для поиска критического значения.
        N: Размер выборки
        verbose: печатает результат КС и критическое значение для выборки.
            По умолчанию True.

    Returns:
        Супремум, который является наибольшей абсолютной разницей между
            найденными и ожидаемыми пропорциями, и критическое значение
            Колмогорова-Смирнова в соответствии с уровнем доверия, для сравнения
    """
    if confidence is None:
        print('\nТест Колмогорова-Смирнова требует уровня доверия, отличного от None.')
        return
    else:
        # сортировка и расчет кумулятивного распределения
        ks_frame = frame.sort_index()[['Факт', 'Теор']].cumsum()
        # поиск супремума - наибольшей кумулятивной разницы
        suprem = ((ks_frame.Факт - ks_frame.Теор).abs()).max()
        # расчет критического значения в соответствии с уровнем доверия
        crit_KS = CRIT_KS[confidence] / sqrt(N)

        if verbose:
            print(f"\nСтатистика Колмогорова-Смирнова равна {suprem:.4f}.\n"
                  f"Критическое КС для этой выборки: {crit_KS:.4f}")
        return (suprem, crit_KS)

def kolmogorov_smirnov_2(frame):
    """Расчет теста Колмогорова-Смирнова для найденных распределений

    Args:
        frame: DataFrame с распределениями Found и Expected.

    Returns:
        Супремум, который является наибольшей абсолютной разницей между
            найденными и ожидаемыми пропорциями
    """
    # сортировка и расчет кумулятивного распределения
    ks_frame = frame.sort_index()[['Факт', 'Теор']].cumsum()
    # поиск супремума - наибольшей кумулятивной разницы
    return ((ks_frame.Факт - ks_frame.Теор).abs()).max()

def _two_dist_ks_(dist1, dist2, cummulative=True):
    """Расчет статистики Колмогорова-Смирнова между двумя распределениями,
    найденным (dist2) и ожидаемым (dist1).

    Args:
        dist1 (np.arrat): массив с ожидаемым распределением
        dist2 (np.array): массив с найденным распределением
        cummulative (bool): применять кумулятивную сумму к распределениям (эмпирическая cdf).

    Returns:
        tuple(floats): статистика КС
    """
    dist2.sort(); dist1.sort()
    if not cummulative:
        return nabs(dist2 - dist1).max()
    return nabs(dist2.cumsum() - dist1.cumsum()).max()

def _mantissas_ks_(mant_dist, confidence, sample_size):
    """Расчет статистики Колмогорова-Смирнова для мантисс, а также
    предоставление критического значения КС в соответствии с предоставленным
    размером выборки и уровнем доверия

    Args:
        mant_dist (np.array): массив с найденным распределением мантисс
        confidence (float, int): уровень доверия для расчета критического значения

    Returns:
        tuple(floats): статистика КС и критическое значение
    """
    crit_ks = CRIT_KS[confidence] * sqrt(2 * sample_size / sample_size ** 2)\
                if confidence else None
    # некумулятивное, равномерно распределенное
    expected = linspace(0, 1, len(mant_dist), endpoint=False)
    ks = _two_dist_ks_(expected, mant_dist, cummulative=False)
    return ks, crit_ks

def mad(frame, test, verbose=True):
    """Расчет среднего абсолютного отклонения (MAD) между найденными и
    ожидаемыми пропорциями.

    Args:
        frame: DataFrame с уже рассчитанными абсолютными отклонениями.
        test: Тест, для которого рассчитывается MAD (F1D, SD, F2D...)
        verbose: печатает результат MAD и сравнивает его с предельными значениями
            соответствия. По умолчанию True.

    Returns:
        Среднее абсолютных отклонений между найденными и ожидаемыми
            пропорциями.
    """
    mad = frame.Абс_Разность.mean()

    if verbose:
        print(f"\nСреднее абсолютное отклонение (MAD) равно {mad}")

        if test != -2:
            print(f"Для {MAD_CONFORM[DIGS[test]]}:\n\
            - 0.0000 до {MAD_CONFORM[test][0]}: Тесное соответствие\n\
            - {MAD_CONFORM[test][0]} до {MAD_CONFORM[test][1]}: Приемлемое соответствие\n\
            - {MAD_CONFORM[test][1]} до {MAD_CONFORM[test][2]}: Соответствие с погрешностью\n\
            - Выше {MAD_CONFORM[test][2]}: Несоответствие")
        else:
            pass
    return mad

def mse(frame, verbose=True):
    """Расчет среднеквадратичной ошибки теста

    Args:
        frame: DataFrame с уже рассчитанными абсолютными отклонениями между
            найденными и ожидаемыми пропорциями
        verbose: Печатает МСО. По умолчанию True.

    Returns:
        Среднее квадратов разниц между найденными и ожидаемыми пропорциями.
    """
    mse = (frame.Абс_Разность ** 2).mean()

    if verbose:
        print(f"\nСреднеквадратичная ошибка = {mse}")

    return mse
def _bhattacharyya_coefficient(dist_1, dist_2):
    """Расчет коэффициента Бхаттачарьи между двумя вероятностными
    распределениями, который будет позже использован для расчета расстояния Бхаттачарьи

    Args:
        dist_1 (np.array): Новое собранное распределение, которое будет сравнено
            с более старым / установленным распределением.
        dist_2 (np.array): Более старое/установленное распределение, с которым
            будет сравнено новое.

    Returns:
        bhat_coef (float)
    """
    return sqrt(dist_1 * dist_2).sum()

def _bhattacharyya_distance_(dist_1, dist_2):
    """Расчет расстояния Бхаттачарьи между двумя вероятностными распределениями

    Args:
        dist_1 (np.array): Новое собранное распределение, которое будет сравнено
            с более старым / установленным распределением.
        dist_2 (np.array): Более старое/установленное распределение, с которым
            будет сравнено новое.

    Returns:
        bhat_dist (float)
    """
    with errstate(divide='ignore'):
        bhat_dist =  -log(_bhattacharyya_coefficient(dist_1, dist_2))
    return bhat_dist

def _kullback_leibler_divergence_(dist_1, dist_2):
    """Расчет дивергенции Куллбака-Лейблера между двумя вероятностными распределениями.

    Args:
        dist_1 (np.array): Новое собранное распределение, которое будет сравнено
            с более старым / установленным распределением.
        dist_2 (np.array): Более старое/установленное распределение, с которым
            будет сравнено новое.

    Returns:
        kulb_leib_diverg (float)
    """
    # игнорирование предупреждения деления на ноль в np.where
    with errstate(divide='ignore'):
        kl_d = (log((dist_1 / dist_2), where=(dist_1 != 0)) * dist_1).sum()
    return kl_d
