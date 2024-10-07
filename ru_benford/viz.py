from numpy import array, arange, maximum, sqrt, ones
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from .constants import COLORS, MAD_CONFORM

def plot_expected(df, digs, save_plot=None, save_plot_kwargs=None):
    """Строит график ожидаемых распределений Бенфорда

    Args:
        df: DataFrame с ожидаемыми пропорциями
        digs: цифра теста
        save_plot: строка с путем/именем файла, в который будет сохранен сгенерированный график. Использует matplotlib.pyplot.savefig(). Формат файла выводится из расширения имени файла.
        save_plot_kwargs: словарь с любыми из аргументов, принимаемых matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    if digs in [1, 2, 3]:
        y_max = (df.Теор.max() + (10 ** -(digs) / 3)) * 100
        figsize = 2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)
    elif digs == 22:
        y_max = 13.
        figsize = 14, 10.5
    elif digs == -2:
        y_max = 1.1
        figsize = 15, 8
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Ожидаемые распределения Бенфорда', size='xx-large')
    plt.xlabel(df.index.name, size='x-large')
    plt.ylabel('Распределение (%)', size='x-large')
    ax.set_facecolor(COLORS['b'])
    ax.set_ylim(0, y_max)
    ax.bar(df.index, df.Теор * 100, color=COLORS['t'], align='center')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index)

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)

def _get_plot_args(digs):
    """Выбирает правильные аргументы для функций построения графиков, в зависимости от выбранного теста (digs).
    """
    if digs in [1, 2, 3]:
        text_x = False
        n, m = 10 ** (digs - 1), 10 ** (digs)
        x = arange(n, m)
        figsize = (2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5))
    elif digs == 22:
        text_x = False
        x = arange(10)
        figsize = (14, 10)
    else:
        text_x = True
        x = arange(100)
        figsize = (15, 7)
    return x, figsize, text_x

def plot_digs(df, x, y_Exp, y_Found, N, figsize, conf_Z, text_x=False,
              save_plot=None, save_plot_kwargs=None):
    """Строит графики результатов тестов цифр

    Args:
        df: DataFrame с данными для построения графика
        x: последовательность, используемая в оси x
        y_Exp: последовательность ожидаемых пропорций, используемых в оси y (линия)
        y_Found: последовательность найденных пропорций, используемых в оси y (столбцы)
        N: длина последовательности, используемая при построении уровней доверия
        figsize: кортеж, задающий размер фигуры графика
        conf_Z: уровень доверия
        save_pic: путь к файлу для сохранения рисунка
        text_x: принудительно показывать все метки оси x. По умолчанию True.
        save_plot: строка с путем/именем файла, в который будет сохранен сгенерированный график. Использует matplotlib.pyplot.savefig(). Формат файла выводится из расширения имени файла.
        save_plot_kwargs: словарь с любыми из аргументов, принимаемых matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    if len(x) > 10:
        rotation = 90
    else:
        rotation = 0
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Теоретическое vs. Фактическое распределение', size='xx-large')
    plt.xlabel('Цифры', size='x-large')
    plt.ylabel('Распределение (%)', size='x-large')
    if conf_Z is not None:
        sig = conf_Z * sqrt(y_Exp * (1 - y_Exp) / N)
        upper = y_Exp + sig + (1 / (2 * N))
        lower_zeros = array([0]*len(upper))
        lower = maximum(y_Exp - sig - (1 / (2 * N)), lower_zeros)
        u = (y_Found < lower) | (y_Found > upper)
        c = array([COLORS['m']] * len(u))
        c[u] = COLORS['af']
        lower *= 100.
        upper *= 100.
        ax.plot(x, upper, color=COLORS['s'], zorder=5)
        ax.plot(x, lower, color=COLORS['s'], zorder=5)
        ax.fill_between(x, upper, lower, color=COLORS['s'],
                        alpha=.3, label='Дов. интервал')
    else:
        c = COLORS['m']
    ax.bar(x, y_Found * 100., color=c, label='Факт.', zorder=3, align='center')
    ax.plot(x, y_Exp * 100., color=COLORS['s'], linewidth=2.5,
            label='Теор.', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=rotation)
    ax.set_facecolor(COLORS['b'])
    if text_x:
        ind = array(df.index).astype(str)
        ind[:10] = array(['00', '01', '02', '03', '04', '05',
                          '06', '07', '08', '09'])
        plt.xticks(x, ind, rotation='vertical')
    ax.legend()
    ax.set_ylim(0, max([y_Exp.max() * 100, y_Found.max() * 100]) + 10 / len(x))
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)

def plot_sum(df, figsize, li, text_x=False, save_plot=None, save_plot_kwargs=None):
    """Строит графики результатов теста суммирования

    Args:
        df: DataFrame с данными для построения графика
        figsize: задает размеры фигуры графика
        li: значение, с которым будет нарисована горизонтальная линия
        save_plot: строка с путем/именем файла, в который будет сохранен сгенерированный график. Использует matplotlib.pyplot.savefig(). Формат файла выводится из расширения имени файла.
        save_plot_kwargs: словарь с любыми из аргументов, принимаемых matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    x = df.index
    rotation = 90 if len(x) > 10 else 0
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.title('Теоретическое vs. Фактическое распределение')
    plt.xlabel('Цифры')
    plt.ylabel('Суммы')
    ax.bar(x, df.Доля, color=COLORS['m'],
           label='Найденные суммы', zorder=3, align='center')
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.axhline(li, color=COLORS['s'], linewidth=2, label='Ожидаемое', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=rotation)
    ax.set_facecolor(COLORS['b'])
    if text_x:
        ind = array(x).astype(str)
        ind[:10] = array(['00', '01', '02', '03', '04', '05',
                          '06', '07', '08', '09'])
        plt.xticks(x, ind, rotation='vertical')
    ax.legend()

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)

def plot_ordered_mantissas(col, figsize=(12, 12),
                           save_plot=None, save_plot_kwargs=None):
    """Строит график упорядоченных мантисс и сравнивает их с ожидаемой прямой линией, которая должна быть сформирована в соответствующем наборе Бенфорда.

    Args:
        col (Series): колонка мантисс для построения графика.
        figsize (tuple): задает размеры фигуры графика.
        save_plot: строка с путем/именем файла, в который будет сохранен сгенерированный график. Использует matplotlib.pyplot.savefig(). Формат файла выводится из расширения имени файла.
        save_plot_kwargs: словарь с любыми из аргументов, принимаемых matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    ld = len(col)
    x = arange(1, ld + 1)
    n = ones(ld) / ld
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(x, col.sort_values(), linestyle='--',
            color=COLORS['s'], linewidth=3, label='Факт.')
    ax.plot(x, n.cumsum(), color=COLORS['m'],
            linewidth=2, label='Ожидаемое')
    plt.ylim((0, 1.))
    plt.xlim((1, ld + 1))
    ax.set_facecolor(COLORS['b'])
    ax.set_title("Упорядоченные мантиссы")
    plt.legend(loc='upper left')

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False);

def plot_mantissa_arc_test(df, gravity_center, grid=True, figsize=12,
                           save_plot=None, save_plot_kwargs=None):
    """Рисует тест дуги мантиссы после вычисления X и Y координат для каждой мантиссы и центра тяжести для набора

    Args:
        df (DataFrame): pandas DataFrame с мантиссами и координатами X и Y.
        gravity_center (tuple): координаты для рисования центра тяжести
        grid (bool): показывать сетку. По умолчанию True.
        figsize (int): размеры фигуры. Не нужно быть кортежем, так как фигура является квадратом.
        save_plot: строка с путем/именем файла, в который будет сохранен сгенерированный график. Использует matplotlib.pyplot.savefig(). Формат файла выводится из расширения имени файла.
        save_plot_kwargs: словарь с любыми из аргументов, принимаемых matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    fig = plt.figure(figsize=(figsize, figsize))
    ax = plt.subplot()
    ax.set_facecolor(COLORS['b'])
    ax.scatter(df.mant_x, df.mant_y, label="Дуговой тест мантисс",
               color=COLORS['m'])
    ax.scatter(gravity_center[0], gravity_center[1],
               color=COLORS['s'])
    text_annotation = Annotation(
        "  Центр тяжести: "
        f"x({round(gravity_center[0], 3)}),"
        f" y({round(gravity_center[1], 3)})",
        xy=(gravity_center[0] - 0.65,
            gravity_center[1] - 0.1),
        xycoords='data')
    ax.add_artist(text_annotation)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.legend(loc='lower left')
    ax.set_title("Дуговой тест мантисс")

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False);

def plot_roll_mse(roll_series, figsize, save_plot=None, save_plot_kwargs=None):
    """Показывает график скользящего MSE

    Args:
        roll_series: pd.Series, полученный из скользящего mse.
        figsize: размеры фигуры.
        save_plot: строка с путем/именем файла, в который будет сохранен сгенерированный график. Использует matplotlib.pyplot.savefig(). Формат файла выводится из расширения имени файла.
        save_plot_kwargs: словарь с любыми из аргументов, принимаемых matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(COLORS['b'])
    ax.plot(roll_series, color=COLORS['m'])

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)

def plot_roll_mad(roll_mad, figsize, save_plot=None, save_plot_kwargs=None):
    """Показывает график скользящего MAD

    Args:
        roll_mad: pd.Series, полученный из скользящего mad.
        figsize: размеры фигуры.
        save_plot: строка с путем/именем файла, в который будет сохранен сгенерированный график. Использует matplotlib.pyplot.savefig(). Формат файла выводится из расширения имени файла.
        save_plot_kwargs: словарь с любыми из аргументов, принимаемых matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(COLORS['b'])
    ax.plot(roll_mad.roll_series, color=COLORS['m'])

    if roll_mad.test != -2:
        plt.axhline(y=MAD_CONFORM[roll_mad.test][0], color=COLORS['af'], linewidth=3)
        plt.axhline(y=MAD_CONFORM[roll_mad.test][1], color=COLORS['h2'], linewidth=3)
        plt.axhline(y=MAD_CONFORM[roll_mad.test][2], color=COLORS['s'], linewidth=3)

    if save_plot:
        if not save_plot_kwargs:
            save_plot_kwargs = {}
        plt.savefig(save_plot, **save_plot_kwargs)

    plt.show(block=False)
