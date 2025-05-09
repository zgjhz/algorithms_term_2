# A B C D E
# 0 3 1 4 8
# 2 0 5 1 3
# 6 2 0 3 7
# 4 3 8 0 2
# 5 6 3 4 0

# A B C D E
# 0 3 0 0 0
# 3 0 2 0 0
# 0 2 0 4 0
# 0 0 4 0 0
# 0 0 0 0 0

# A B C D E F G H I J K L M N O
# 0  3  17 16 20  7  2  1  5  3 15 17 14 16 19
# 14 0  17 17 11 13 11 20 4  17 8  9  18 20 11
# 8  4  0  15 12 14 18 18 5  7  9  3  11 15 12
# 5  7  16 0  18 8  15 10 8  5  11 13 15 11 17
# 13 13 11 20 0  16 6  9  1  9  15 5  1  15 10
# 2  17 11 12 15 0  10 9  1  6  6  18 15 6  15
# 4  18 9  14 13 13 0  3  16 8  6  2  6  19 15
# 12 18 15 11 6  19 5  0  6  4  17 12 19 3  12
# 8  12 16 2  18 5  7  15 0  7  7  2  1  6  4
# 11 12 3  15 8  18 18 8  17 0  12 10 9  17 20
# 8  3  20 14 18 19 8  18 18 2  0  15 5  2  13
# 13 6  20 2  4  19 10 8  19 8  8  0  17 7  1
# 11 9  4  11 14 5  10 20 7  2  1  16 0  18 4
# 16 9  12 19 14 10 2  2  20 5  1  5  20 0  1
# 10 12 15 19 14 17 5  14 15 1  6  16 1  15 0


import sys
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QCheckBox, QTextEdit, QComboBox, QInputDialog
from PyQt5.QtGui import QPainter, QPen, QBrush, QPolygonF
from PyQt5.QtCore import Qt, QPointF

cities = []
distance_matrix = np.array([])

def parse_graph_input(graph_text):
    global cities, distance_matrix
    lines = graph_text.strip().split("\n")
    cities = lines[0].split()
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    
    for i in range(1, len(lines)):
        values = list(map(int, lines[i].split()))
        for j in range(n):
            distance_matrix[i - 1][j] = values[j]

def nearest_neighbor_tsp(matrix, start=0, optimization=False):
    n = len(matrix)
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    for _ in range(n - 1):
        last = path[-1]
        nearest = np.inf
        nearest_index = -1
        
        for i in range(n):
            if not visited[i] and matrix[last][i] > 0 and matrix[last][i] < nearest:
                nearest = matrix[last][i]
                nearest_index = i
        
        if nearest_index == -1:
            return [], 0
        
        path.append(nearest_index)
        visited[nearest_index] = True
    
    if matrix[path[-1]][start] > 0:
        path.append(start)
    else:
        return [], 0
    
    path_length = sum(matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
    
    if optimization:
        best_path = path
        best_cost = path_length
        
        for new_start in range(n):
            if new_start == start:
                continue
            new_path, new_cost = nearest_neighbor_tsp(matrix, start=new_start)
            if new_path and new_cost < best_cost:
                best_path, best_cost = new_path, new_cost
        
        return best_path, best_cost
    
    return path, path_length

def draw_graph(path):
    if not path:
        print("Невозможно найти путь, возвращающийся в начальный город.")
        return
    
    G = nx.DiGraph()
    for i in range(len(cities)):
        for j in range(len(cities)):
            if distance_matrix[i][j] > 0:
                G.add_edge(cities[i], cities[j], weight=distance_matrix[i][j])
    
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    path_edges = [(cities[path[i]], cities[path[i + 1]]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, arrows=True)
    plt.show()

class GraphEditor(QWidget):
    def __init__(self, tsp_app):
        super().__init__()
        self.setWindowTitle("Редактор графа")
        self.resize(500, 500)
        self.nodes = []  # Список вершин
        self.edges = []  # Список рёбер (начальная вершина, конечная вершина, вес)
        self.tsp_app = tsp_app
        self.start_node = None  # Временное хранилище для соединения вершин

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            for i, (x, y) in enumerate(self.nodes):
                if (x - pos.x()) ** 2 + (y - pos.y()) ** 2 <= 400:  # Проверка нажатия на вершину
                    if self.start_node is None:
                        self.start_node = i  # Выбираем начальную вершину
                    else:
                        weight, ok = QInputDialog.getInt(self, "Введите вес", "Вес ребра:", min=1)
                        if ok:
                            self.edges.append((self.start_node, i, weight))
                        self.start_node = None  # Сброс выбора
                    self.update()
                    return
            
            # Добавляем новую вершину
            self.nodes.append((pos.x(), pos.y()))
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Рисуем рёбра со стрелками
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        for start, end, weight in self.edges:
            x1, y1 = self.nodes[start]
            x2, y2 = self.nodes[end]
            painter.drawLine(x1, y1, x2, y2)
            self.draw_arrow(painter, x1, y1, x2, y2)
            painter.drawText(int((x1 + x2) / 2), int((y1 + y2) / 2), str(weight))
        
        # Рисуем вершины
        painter.setBrush(QBrush(Qt.blue))
        for i, (x, y) in enumerate(self.nodes):
            painter.drawEllipse(QPointF(x, y), 10, 10)
            painter.drawText(int(x - 10), int(y - 10), str(i))

    def draw_arrow(self, painter, x1, y1, x2, y2):
        arrow_size = 20
        angle = np.arctan2(y2 - y1, x2 - x1)
        
        line_pen = QPen(Qt.black, 2)
        painter.setPen(line_pen)
        painter.drawLine(x1, y1, x2, y2)
        
        p1 = QPointF(x2 - arrow_size * np.cos(angle - np.pi / 6), y2 - arrow_size * np.sin(angle - np.pi / 6))
        p2 = QPointF(x2, y2)
        p3 = QPointF(x2 - arrow_size * np.cos(angle + np.pi / 6), y2 - arrow_size * np.sin(angle + np.pi / 6))
        
        arrow_head = QPolygonF([p1, p2, p3])
        painter.setBrush(QBrush(Qt.yellow))
        painter.drawPolygon(arrow_head)

    def get_graph_data(self):
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        for start, end, weight in self.edges:
            matrix[start][end] = weight
        return [str(i) for i in range(n)], matrix

class TSPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel("Введите граф (первая строка - города, далее матрица расстояний):")
        self.graph_input = QTextEdit()
        self.draw_graph_button = QPushButton("Создать граф вручную")
        self.draw_graph_button.clicked.connect(self.open_graph_editor)
        self.add_graph_button = QPushButton("Добавить граф")
        self.add_graph_button.clicked.connect(self.load_graph)
        self.optimization_checkbox = QCheckBox("Включить модификацию")
        self.start_node_label = QLabel("Выберите начальный город:")
        self.start_node_selector = QComboBox()
        self.button = QPushButton("Найти путь")
        self.button.clicked.connect(self.solve_tsp)
        self.result_label = QLabel("")
        self.path_label = QLabel("")
        self.path_length_label = QLabel("")
        
        layout.addWidget(self.label)
        layout.addWidget(self.graph_input)
        layout.addWidget(self.draw_graph_button)
        layout.addWidget(self.add_graph_button)
        layout.addWidget(self.start_node_label)
        layout.addWidget(self.start_node_selector)
        layout.addWidget(self.optimization_checkbox)
        layout.addWidget(self.button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.path_label)
        layout.addWidget(self.path_length_label)
        self.setLayout(layout)
    
    def load_graph(self):
        global cities, distance_matrix
        graph_text = self.graph_input.toPlainText()
        parse_graph_input(graph_text)
        if hasattr(self, 'graph_editor') and self.graph_editor.nodes:
            cities, distance_matrix = self.graph_editor.get_graph_data()
        self.start_node_selector.clear()
        self.start_node_selector.addItems(cities)
    
    def open_graph_editor(self):
        self.graph_editor = GraphEditor(self)
        self.graph_editor.show()

    def generate_random_graph(self, num_nodes, max_distance=100):
        matrix = np.random.randint(1, max_distance, size=(num_nodes, num_nodes))
        np.fill_diagonal(matrix, 0)
        return matrix

    def test_algorithm(self):
        node_sizes = [6, 10, 15, 30]
        results = []
        
        for n in node_sizes:
            matrix = self.generate_random_graph(n)
            
            start_time = time.time()
            path_normal, cost_normal = nearest_neighbor_tsp(matrix, optimization=False)
            time_normal = time.time() - start_time
            
            start_time = time.time()
            path_cauchy, cost_cauchy = nearest_neighbor_tsp(matrix, optimization=True)
            time_cauchy = time.time() - start_time
            
            results.append((n, cost_normal, time_normal, cost_cauchy, time_cauchy))
        
        print("\nРазмер графа | Стоимость (обычный) | Время (обычный) | Стоимость (Коши) | Время (Коши)")
        for res in results:
            print(f"{res[0]:<12} | {res[1]:<20} | {res[2]:<15.5f} | {res[3]:<18} | {res[4]:<15.5f}")
    
    def solve_tsp(self):
        use_optimization = self.optimization_checkbox.isChecked()
        start_index = self.start_node_selector.currentIndex()
        
        self.test_algorithm()
        path, path_length = nearest_neighbor_tsp(distance_matrix, start=start_index, optimization=use_optimization)
        
        if path:
            formatted_path = ' → '.join([cities[i] for i in path])
            self.result_label.setText(f"Оптимальный путь: {formatted_path}")
            # self.path_label.setText(f"Путь: {' → '.join([cities[i] for i in path])}")
            self.path_length_label.setText(f"Длина пути: {path_length}")
            draw_graph(path)
        else:
            self.result_label.setText("Невозможно построить цикл, возвращающийся в начальный город.")
            self.path_label.setText("")
            self.path_length_label.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TSPApp()
    ex.setWindowTitle("Задача коммивояжёра (Ориентированный граф)")
    ex.resize(400, 400)
    ex.show()
    sys.exit(app.exec_())