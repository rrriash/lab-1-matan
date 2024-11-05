import numpy as np
import matplotlib.pyplot as plt

# Определим функцию для x_n
def x_n(n):
    return np.arctan((1 + (-1)**n) / 2) * ((20*n + 3) / (n + 2)) # Формула для x_n

# Генерация первых 100 значений последовательности x_n
n_values = np.arange(1, 101)  # Массив n от 1 до 100
x_values = np.array([x_n(n) for n in n_values])  # Массив значений x_n для каждого n

# Расчет sup, inf, limsup и liminf
sup_x = np.max(x_values)  # Верхняя граница (максимум всех значений)
inf_x = np.min(x_values)  # Нижняя граница (минимум всех значений)
limsup_x = np.max(x_values[::2])  # Пики на четных n (верхняя граница для четных)
liminf_x = np.min(x_values[::2])  # Пики на нечетных n (нижняя граница для нечетных)

# Выделение сходящейся подпоследовательности для четных n (например)
even_n_values = n_values[::2]  # Индексы для четных значений n
even_x_values = x_values[::2]  # Значения x_n для четных n

# Построение графика
plt.figure(figsize=(10, 6))  # Настройка размера графика

# Строим график всей последовательности
plt.plot(n_values, x_values, label='Полная последовательность $x_n$', marker='o', color='b')  # График всей последовательности

# Добавление точек для сходящейся подпоследовательности (четные n) другим цветом
plt.scatter(even_n_values, even_x_values, color='purple', label='Сходящаяся подпоследовательность (четные n)', zorder=5)

# Добавление горизонтальных линий для sup, inf, limsup, liminf
plt.axhline(y=sup_x, color='r', linestyle='--', label=f'$sup(x_n) = {sup_x:.2f}$')  # Линия для sup
plt.axhline(y=inf_x, color='g', linestyle='--', label=f'$inf(x_n) = {inf_x:.2f}$')  # Линия для inf
plt.axhline(y=limsup_x, color='orange', linestyle='--', label=f'$limsup(x_n) = {limsup_x:.2f}$')  # Линия для limsup
plt.axhline(y=liminf_x, color='purple', linestyle='--', label=f'$liminf(x_n) = {liminf_x:.2f}$')  # Линия для liminf

# Настройки графика
plt.title('График последовательности $x_n$ с выделением сходящейся подпоследовательности')
plt.xlabel('$n$')
plt.ylabel('$x_n$')
plt.legend(loc='best')  # Легенда
plt.grid(True)  # Включаем сетку

# Показать график
plt.show()  # Это обязательно для отображения графика

#код номер 2
import numpy as np
import matplotlib.pyplot as plt

# Определим функцию для x_n
def x_n(n):
    return np.arctan((1 + (-1)**n) / 2) * ((20*n + 3) / (n + 2)) # Формула для x_n

# Генерация первых 100 значений последовательности x_n
n_values = np.arange(1, 101)  # Массив n от 1 до 100
x_values = np.array([x_n(n) for n in n_values])  # Массив значений x_n для каждого n

# Рассчитываем предел для последовательности
# Предполагаем, что предел существует, и ищем его как лимит для больших n
lim_x = np.mean(x_values[90:])  # Среднее значение для больших n, как приближенный предел

# Настроим точность для окрестности предела
epsilon = 0.01  # Погрешность, например, 0.01

# Ищем номер n0, начиная с которого члены подпоследовательности попадают в ε-окрестность
indices = np.where(np.abs(x_values - lim_x) < epsilon)[0]  # Получаем индексы, где условие выполняется

if indices.size > 0:
    n0 = indices[0] + 1  # n0 - это первое значение, соответствующее условию, +1 для учёта индексации с 1
else:
    n0 = None  # Если нет элементов, попадающих в окрестность, устанавливаем n0 в None

# Построение графика
plt.figure(figsize=(10, 6))  # Настройка размера графика

# Строим график всей последовательности
plt.plot(n_values, x_values, label='Полная последовательность $x_n$', marker='o', color='b')  # График всей последовательности

# Добавление точек для подпоследовательности, начиная с найденного n0
if n0 is not None:
    subseq_n_values = n_values[n0-1:]  # Индексы с n0
    subseq_x_values = x_values[n0-1:]  # Значения x_n с n0
    plt.scatter(subseq_n_values, subseq_x_values, color='purple', label=f'Подпоследовательность (с $n_0 = {n0}$)', zorder=5)
else:
    print("Нет значений, попадающих в ε-окрестность предела.")

# Добавление горизонтальной линии для предела
plt.axhline(y=lim_x, color='r', linestyle='--', label=f'$lim(x_n) = {lim_x:.2f}$')

# Настройки графика
plt.title('График последовательности $x_n$ с выделением сходящейся подпоследовательности')
plt.xlabel('$n$')
plt.ylabel('$x_n$')
plt.legend(loc='best')  # Легенда
plt.grid(True)  # Включаем сетку

# Показать график
plt.show()  # Это обязательно для отображения графика

# Выводим номер n0 и найденный предел
if n0 is not None:
    print(f"Номер n0: {n0}")
    print(f"Приближенный предел: {lim_x:.4f}")
else:
    print("Не удалось найти номер n0, так как последовательность не попадает в ε-окрестность.")

#код 3
import numpy as np
import matplotlib.pyplot as plt

# Определим функцию для x_n
def x_n(n):
    return np.arctan((1 + (-1)**n) / 2) * ((20*n + 3) / (n + 2))  # Формула для x_n

# Генерация первых 100 значений последовательности x_n
n_values = np.arange(1, 101)  # Массив n от 1 до 100
x_values = np.array([x_n(n) for n in n_values])  # Массив значений x_n для каждого n

# Расчет sup и inf
sup_x = np.max(x_values)  # Верхняя граница (максимум всех значений)
inf_x = np.min(x_values)  # Нижняя граница (минимум всех значений)

# Параметр ε
epsilon = 0.01  # Например, ε = 0.01

# Нахождение номера m, такого что x_m > sup(x_n) - ε
m_sup = np.min(np.where(x_values > sup_x - epsilon))  # Первый индекс, где x_m > sup_x - ε

# Нахождение номера m, такого что x_m < inf(x_n) + ε
m_inf = np.min(np.where(x_values < inf_x + epsilon))  # Первый индекс, где x_m < inf_x + ε

# Построение графика
plt.figure(figsize=(10, 6))  # Настройка размера графика

# Строим график всей последовательности
plt.plot(n_values, x_values, label='Полная последовательность $x_n$', marker='o', color='b')  # График всей последовательности

# Добавление горизонтальных линий для sup и inf
plt.axhline(y=sup_x, color='r', linestyle='--', label=f'$sup(x_n) = {sup_x:.2f}$')  # Линия для sup
plt.axhline(y=inf_x, color='g', linestyle='--', label=f'$inf(x_n) = {inf_x:.2f}$')  # Линия для inf

# Отметим найденные точки на графике
plt.scatter(n_values[m_sup], x_values[m_sup], color='orange', label=f'Точка m = {m_sup} для sup(x_n) - ε', zorder=5)
plt.scatter(n_values[m_inf], x_values[m_inf], color='purple', label=f'Точка m = {m_inf} для inf(x_n) + ε', zorder=5)

# Настройки графика
plt.title('График последовательности $x_n$ с точками для границ')
plt.xlabel('$n$')
plt.ylabel('$x_n$')
plt.legend(loc='best')  # Легенда
plt.grid(True)  # Включаем сетку

# Показать график
plt.show()  # Это обязательно для отображения графика

# Выводим номера m для sup и inf
print(f"Номер m для sup(x_n) - ε: {m_sup}")
print(f"Номер m для inf(x_n) + ε: {m_inf}")