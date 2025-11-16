#include "visualizer.h"
#include "random_variable.h"
#include "random_walk_visualizer.h"
#include <iostream>
#include <limits>
#include <windows.h>
#include <commdlg.h>
#include <set>
#include <fstream>
#include <string>

namespace {
	std::vector<std::pair<double, double>> readDistributionFromStdin() {
		std::vector<std::pair<double, double>> dist;
		std::cout << "Введите количество исходов n: ";
		int n = 0;
		while (!(std::cin >> n) || n <= 0) {
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "Некорректное n. Повторите: ";
		}

		std::set<double> usedValues;
		dist.reserve(static_cast<size_t>(n));
		for (int i = 0; i < n; ++i) {
			double x = 0.0, p = 0.0;
			std::cout << "Введите пару x p для i=" << (i + 1) << ": ";
			while (!(std::cin >> x >> p) || p < 0.0) {
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				std::cout << "Некорректный ввод. Ожидается: <x: число> <p: неотрицательное>. Повторите: ";
			}
			if (usedValues.count(x)) {
				std::cout << "Значение x=" << x << " уже встречалось. Введите уникальное значение.\n";
				--i;
				continue;
			}
			usedValues.insert(x);
			dist.emplace_back(x, p);
		}

		double sum = 0.0;
		for (const auto &pr : dist) sum += pr.second;
		const double eps = 1e-9;
		if (sum <= eps) {
			throw std::runtime_error("Сумма вероятностей равна нулю.");
		}
		if (std::abs(sum - 1.0) > 1e-6) {
			std::cout << "Сумма вероятностей = " << sum << ", будет выполнена нормализация к 1.\n";
			for (auto &pr : dist) pr.second /= sum;
		}
		return dist;
	}

	void demonstrateTask1_1() {
		std::cout << "=== Задание 1.1: Демонстрация функционала DiscreteRandomVariable ===\n\n";

		// Выбор источника распределения: CLI или пример по умолчанию
		bool useCli = false;
		std::cout << "Использовать ввод с клавиатуры? (y/n): ";
		char choice;
		std::cin >> choice;
		if (choice == 'y' || choice == 'Y') {
			useCli = true;
		}

		DiscreteRandomVariable rv1;
		if (useCli) {
			try {
				auto dist = readDistributionFromStdin();
				rv1.setDistribution(dist);
			} catch (const std::exception& ex) {
				std::cerr << "Ошибка ввода распределения: " << ex.what() << "\n";
				return;
			}
		} else {
			std::vector<std::pair<double, double>> dist1 = {
				{1, 0.1}, {2, 0.2}, {3, 0.4}, {4, 0.2}, {5, 0.1}
			};
			rv1.setDistribution(dist1);
			std::cout << "Используется распределение по умолчанию: {1:0.1, 2:0.2, 3:0.4, 4:0.2, 5:0.1}\n\n";
		}

		// 1. Отображение в виде закона распределения
		std::cout << "1. Закон распределения:\n";
		std::cout << rv1.toString() << "\n";

		// 2. Отображение в виде полилайна
		std::cout << "2. Отображение в виде полилайна:\n";
		std::cout << rv1.toPolylineString() << "\n";

		// 3. Отображение в виде функции распределения
		std::cout << "3. Отображение в виде функции распределения:\n";
		std::cout << rv1.toCDFString() << "\n";

		// 4. Умножение на скаляр
		std::cout << "4. Умножение ДСВ на скаляр (X * 2.0):\n";
		DiscreteRandomVariable rv2 = rv1 * 2.0;
		std::cout << rv2.toString() << "\n";

		// 5. Сложение ДСВ
		std::cout << "5. Сложение ДСВ (X + X):\n";
		DiscreteRandomVariable rv3 = rv1 + rv1;
		std::cout << rv3.toString() << "\n";

		// 6. Умножение ДСВ
		std::cout << "6. Умножение ДСВ (X * X):\n";
		DiscreteRandomVariable rv4 = rv1 * rv1;
		std::cout << rv4.toString() << "\n";

		// 7. Статистические характеристики
		std::cout << "7. Статистические характеристики исходной ДСВ:\n";
		std::cout << "   Математическое ожидание: " << rv1.expectation() << "\n";
		std::cout << "   Дисперсия: " << rv1.variance() << "\n";
		std::cout << "   Среднеквадратическое отклонение: " << rv1.standardDeviation() << "\n";
		std::cout << "   Коэффициент асимметрии: " << rv1.skewness() << "\n";
		std::cout << "   Коэффициент эксцесса: " << rv1.kurtosis() << "\n\n";

		// 8. Сериализация и десериализация
		std::cout << "8. Сериализация и десериализация:\n";
		const std::string filename = "rv_serialized.bin";
		try {
			// Сериализация
			std::ofstream outFile(filename, std::ios::binary);
			if (!outFile.is_open()) {
				throw std::runtime_error("Не удалось открыть файл для записи");
			}
			rv1.serialize(outFile);
			outFile.close();
			std::cout << "   ДСВ сериализована в файл: " << filename << "\n";

			// Десериализация
			std::ifstream inFile(filename, std::ios::binary);
			if (!inFile.is_open()) {
				throw std::runtime_error("Не удалось открыть файл для чтения");
			}
			DiscreteRandomVariable rv5;
			rv5.deserialize(inFile);
			inFile.close();
			std::cout << "   ДСВ десериализована из файла: " << filename << "\n";
			std::cout << "   Проверка (должно совпадать с исходной):\n";
			std::cout << rv5.toString() << "\n";
		} catch (const std::exception& ex) {
			std::cerr << "   Ошибка при сериализации/десериализации: " << ex.what() << "\n";
		}

		std::cout << "\n=== Задание 1.1 завершено ===\n";
	}

	void demonstrateTask1_2() {
		std::cout << "=== Задание 1.2: Визуализация ДСВ через OpenGL ===\n\n";

		Visualizer visualizer;

		// Выбор источника распределения: CLI или пример по умолчанию
		bool useCli = false;
		std::cout << "Использовать ввод с клавиатуры? (y/n): ";
		char choice;
		std::cin >> choice;
		if (choice == 'y' || choice == 'Y') {
			useCli = true;
		}

		DiscreteRandomVariable rv1;
		if (useCli) {
			try {
				auto dist = readDistributionFromStdin();
				rv1.setDistribution(dist);
			} catch (const std::exception& ex) {
				std::cerr << "Ошибка ввода распределения: " << ex.what() << "\n";
				return;
			}
		} else {
			std::vector<std::pair<double, double>> dist1 = {
				{1, 0.1}, {2, 0.2}, {3, 0.4}, {4, 0.2}, {5, 0.1}
			};
			rv1.setDistribution(dist1);
			std::cout << "Используется распределение по умолчанию: {1:0.1, 2:0.2, 3:0.4, 4:0.2, 5:0.1}\n\n";
		}

		// Инициализация окна
		if (!visualizer.initialize(800, 600, "Discrete Random Variable Visualizer")) {
			std::cerr << "Ошибка инициализации визуализатора\n";
			return;
		}

		// Устанавливаем указатель на визуализатор для callback'ов
		glfwSetWindowUserPointer(visualizer.getWindow(), &visualizer);

		// Установка для визуализации
		visualizer.setRandomVariable(rv1);

		std::cout << "Controls:\n";
		std::cout << "SPACE - Change view mode\n";
		std::cout << "ESC - Exit\n\n";

		visualizer.run();

		std::cout << "\n=== Задание 1.2 завершено ===\n";
	}

	std::string openFileDialog() {
		OPENFILENAMEA ofn;
		char szFile[260] = { 0 };
		
		ZeroMemory(&ofn, sizeof(ofn));
		ofn.lStructSize = sizeof(ofn);
		ofn.lpstrFile = szFile;
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrFilter = "Binary Files\0*.bin\0All Files\0*.*\0";
		ofn.nFilterIndex = 1;
		ofn.lpstrFileTitle = NULL;
		ofn.nMaxFileTitle = 0;
		ofn.lpstrInitialDir = NULL;
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
		
		if (GetOpenFileNameA(&ofn) == TRUE) {
			return std::string(szFile);
		}
		return "";
	}

	void demonstrateTask2() {
		std::cout << "=== Задание 2: Моделирование случайного блуждания ===\n\n";

		RandomWalkVisualizer visualizer;

		// Инициализация окна
		if (!visualizer.initialize(1000, 600, "Random Walk Simulator")) {
			std::cerr << "Ошибка инициализации визуализатора\n";
			return;
		}

		// Устанавливаем указатель на визуализатор для callback'ов
		glfwSetWindowUserPointer(visualizer.getWindow(), &visualizer);

		// Параметры по умолчанию
		double initialPosition = 0.0;
		int numberOfSteps = 10;
		DiscreteRandomVariable stepDistribution;

		// Выбор источника распределения
		std::cout << "Выберите способ задания закона перемещения:\n";
		std::cout << "1 - Использовать распределение по умолчанию\n";
		std::cout << "2 - Загрузить из файла (десериализация)\n";
		std::cout << "3 - Ввести с клавиатуры\n";
		std::cout << "Ваш выбор: ";

		int choice;
		std::cin >> choice;

		if (choice == 2) {
			// Загрузка из файла
			std::cout << "\nОткрывается диалог выбора файла...\n";
			std::string filename = openFileDialog();
			if (!filename.empty()) {
				visualizer.loadStepDistributionFromFile(filename);
			} else {
				std::cout << "Файл не выбран, используется распределение по умолчанию.\n";
				std::vector<std::pair<double, double>> defaultDist = {
					{-1.0, 0.25}, {0.0, 0.5}, {1.0, 0.25}
				};
				stepDistribution.setDistribution(defaultDist);
				visualizer.setStepDistribution(stepDistribution);
			}
		} else if (choice == 3) {
			// Ввод с клавиатуры
			try {
				auto dist = readDistributionFromStdin();
				stepDistribution.setDistribution(dist);
				visualizer.setStepDistribution(stepDistribution);
			} catch (const std::exception& ex) {
				std::cerr << "Ошибка ввода распределения: " << ex.what() << "\n";
				std::cout << "Используется распределение по умолчанию.\n";
				std::vector<std::pair<double, double>> defaultDist = {
					{-1.0, 0.25}, {0.0, 0.5}, {1.0, 0.25}
				};
				stepDistribution.setDistribution(defaultDist);
				visualizer.setStepDistribution(stepDistribution);
			}
		} else {
			// По умолчанию
			std::vector<std::pair<double, double>> defaultDist = {
				{-1.0, 0.25}, {0.0, 0.5}, {1.0, 0.25}
			};
			stepDistribution.setDistribution(defaultDist);
			visualizer.setStepDistribution(stepDistribution);
			std::cout << "Используется распределение по умолчанию: {-1:0.25, 0:0.5, 1:0.25}\n";
		}

		// Ввод начальной позиции
		std::cout << "\nВведите начальное положение точки (по умолчанию 0.0): ";
		std::string input;
		std::cin.ignore();
		std::getline(std::cin, input);
		if (!input.empty()) {
			try {
				initialPosition = std::stod(input);
			} catch (...) {
				initialPosition = 0.0;
			}
		}
		visualizer.setInitialPosition(initialPosition);

		// Ввод количества шагов
		std::cout << "Введите количество шагов (по умолчанию 10): ";
		std::getline(std::cin, input);
		if (!input.empty()) {
			try {
				numberOfSteps = std::stoi(input);
				if (numberOfSteps <= 0) numberOfSteps = 10;
			} catch (...) {
				numberOfSteps = 10;
			}
		}
		visualizer.setNumberOfSteps(numberOfSteps);

		std::cout << "\n=== Управление ===\n";
		std::cout << "SPACE - Запуск/Пауза\n";
		std::cout << "R - Сброс\n";
		std::cout << "S - Остановка\n";
		std::cout << "D - Показать/скрыть распределение конечных позиций\n";
		std::cout << "ESC - Выход\n\n";

		visualizer.run();

		// Вывод результата после завершения
		std::cout << "\n=== Результат моделирования ===\n";
		std::cout << "Распределение вероятностей попадания точки во все конечные позиции:\n";
		auto& walk = visualizer.getWalk();
		if (walk.isCompleted()) {
			std::cout << walk.getFinalPositionDistributionString() << "\n";
		} else {
			std::cout << "Моделирование не завершено.\n";
		}

		std::cout << "\n=== Задание 2 завершено ===\n";
	}
}

int main(int argc, char** argv) {
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);

	// Парсинг аргументов командной строки
	if (argc < 2) {
		std::cerr << "Использование: " << argv[0] << " <номер_задания> [опции]\n";
		std::cerr << "Примеры:\n";
		std::cerr << "  " << argv[0] << " 1.1  - Демонстрация функционала DiscreteRandomVariable\n";
		std::cerr << "  " << argv[0] << " 1.2  - Визуализация ДСВ через OpenGL\n";
		std::cerr << "  " << argv[0] << " 2    - Моделирование случайного блуждания\n";
		return -1;
	}

	std::string taskNumber = argv[1];

	if (taskNumber == "1.1") {
		demonstrateTask1_1();
	} else if (taskNumber == "1.2") {
		demonstrateTask1_2();
	} else if (taskNumber == "2") {
		demonstrateTask2();
	} else {
		std::cerr << "Неизвестный номер задания: " << taskNumber << "\n";
		std::cerr << "Доступные задания: 1.1, 1.2, 2\n";
		return -1;
	}

	return 0;
}