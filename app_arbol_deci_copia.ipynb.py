import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier




# Cargar el DataFrame desde el archivo Excel
def load_data(file_path):
    dataframe = pd.read_excel(file_path)
    return dataframe

# Preprocesamiento de datos
def preprocess_data(data, specific_labels):
    data = data[data['Sector del establecimento'].isin(specific_labels)]
    data = data.dropna()
    return data

# Entrenar el modelo de árbol de decisiones
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Función para trazar el gráfico de barras
def plot_bar_chart(data):
    plt.figure(figsize=(9, 5))
    sns.countplot(x='Provincia de ubicación', hue='Predicciones', data=data, palette='Set1')
    plt.title("Distribución de Clases por Provincia")
    plt.xlabel("Provincia")
    plt.ylabel("Cantidad")
    plt.legend(title='Clase')
    plt.xticks(rotation=90)
    plt.show()

# Función para filtrar y visualizar datos por provincia
def filter_data_and_plot(data, province, X_train, model, predictors):
    filtered_data = data[data['Provincia de ubicación'] == province]
    X_filtered = pd.get_dummies(filtered_data[predictors], columns=['Provincia de ubicación'], drop_first=True)
    X_filtered = X_filtered.reindex(columns=X_train.columns, fill_value=0)
    y_pred_filtered = model.predict(X_filtered)

    X_filtered_with_predictions = X_filtered.copy()
    X_filtered_with_predictions['Predicciones'] = y_pred_filtered
    X_filtered_with_predictions['Provincia de ubicación'] = filtered_data['Provincia de ubicación']

    plot_bar_chart(X_filtered_with_predictions)


def main():
    specific_labels = ['Privado con Fines de Lucro', 'Público', 'Privado sin Fines de Lucro']
    file_path = 'C:/Users/henry/tesis/data_FINAL_ETIQUETADO.xlsx'
    
    dataframe = load_data(file_path)
    dataframe = preprocess_data(dataframe, specific_labels)
    
    predictors = [
        
        'Medicina interna Dotación Normal total',
    'Medicina interna Disponibles',
    'Cirugía Dotación Normal total',
    'Cirugía Disponibles',
    'Ginecología y Obstetricia Dotación Normal total',
    'Ginecología y Obstetricia Disponibles',
    'Pediatría Dotación Normal total',
    'Pediatría Disponibles',
    'Cardiología Dotación Normal total',
    'Cardiología Disponibles',
    'Neumología Dotación Normal total',
    'Neumología Disponibles',
    'Psiquiatría Dotación Normal total',
    'Psiquiatría Disponibles',
    'Traumatología Dotación Normal total',
    'Traumatología Disponibles',
    'Infectología Dotación Normal total',
    'Infectología Disponibles',
    'Urología Dotación Normal total',
    'Urología Disponibles',
    'Gastroenterología Dotación Normal total',
    'Gastroenterología Disponibles',
    'Servicios Indiferenciados Dotación Normal total',
    'Servicios Indiferenciados Disponibles',
    'Total Dotación normal',
    'Total Camas Disponibles',
    'Camas de emergencia',
    'Camas de cuidados intensivos adultos',
    'Dias de estada',
    'Días-cama disponibles',
    'Fallecidos en menos de 48 horas',
    'Fallecidos en más de 48 horas',
    'Provincia de ubicación'
    ]
    
    
    target = 'Sector del establecimento'
    
    X_filtered = pd.get_dummies(dataframe[predictors], columns=['Provincia de ubicación'], drop_first=True)
    
    y = dataframe[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
    
    model = train_decision_tree(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    X_test_with_predictions = X_test.copy()
    X_test_with_predictions['Predicciones'] = y_pred
    X_test_with_predictions['Provincia de ubicación'] = dataframe['Provincia de ubicación']
    
    while True:
        from colorama import Fore, Back, init
        init (autoreset=True)
        print (Fore.GREEN + "-----------------FORMULARIO----------------------")
        print(Fore.RED + "\n Selecciona una opción:\n")
        print(Fore.BLUE+"1. Mostrar gráfico de distribución de clases por provincia")
        print("\n")
        print(Fore.BLUE+"2. Filtrar por provincia y mostrar gráfico")
        print("\n")
        print(Fore.BLUE+"3. Salir")
        print("\n")
        
        option = input("Ingresa el número de la opción: ")
        
        if option == '1':
            plot_bar_chart(X_test_with_predictions)
        elif option == '2':
            unique_provinces = dataframe['Provincia de ubicación'].unique()
            print("Provincias disponibles:")
            for idx, province in enumerate(unique_provinces, start=1):
                print(f"{idx}. {province}")
            
            province_numbers = input("Ingrese el numeral de la provincia: ")
            selected_numbers = [int(num.strip()) for num in province_numbers.split(',')]
            
            for num in selected_numbers:
                if 1 <= num <= len(unique_provinces):
                    province = unique_provinces[num - 1]
                    filter_data_and_plot(dataframe, province, X_train, model, predictors)

                else:
                    print(f"Número inválido: {num}. Ignorando.")
        elif option == '3':
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida. Por favor, selecciona una opción válida.")

if __name__ == "__main__":
    main()
