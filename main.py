"""
# Codificación de Variables Categóricas para Machine Learning
Este script demuestra diferentes técnicas para manejar variables categóricas en proyectos de machine learning,
incluyendo la identificación de columnas seguras para codificación, análisis de cardinalidad, y ejemplos
de codificación ordinal y one-hot.

Autor: Ronald Vilcas
GitHub: ronaldvm7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# Configuración para visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


def crear_datos_ejemplo():
    """
    Crea un conjunto de datos de ejemplo con variables categóricas
    de diferentes cardinalidades para demostrar los conceptos.
    """
    np.random.seed(42)
    n_samples = 1000

    # Crear columnas categóricas con diferente cardinalidad
    colores = np.random.choice(['rojo', 'verde', 'azul', 'amarillo', 'negro'], n_samples)
    tamaños = np.random.choice(['pequeño', 'mediano', 'grande'], n_samples)

    # Columna de alta cardinalidad (50 valores únicos)
    categorias_altas = [f'categoria_{i}' for i in range(50)]
    alta_cardinalidad = np.random.choice(categorias_altas, n_samples)

    # Variable objetivo (para ejemplo de modelo)
    target = np.random.randint(0, 2, n_samples)

    # Crear DataFrame
    data = pd.DataFrame({
        'color': colores,
        'tamaño': tamaños,
        'categoria': alta_cardinalidad,
        'target': target
    })

    # Agregar una columna numérica para completar el ejemplo
    data['valor_numerico'] = np.random.normal(0, 1, n_samples)

    return data


def visualizar_cardinalidad(df):
    """
    Visualiza la cardinalidad (número de valores únicos) de las columnas categóricas.
    """
    # Identificar columnas categóricas
    object_cols = [col for col in df.columns if df[col].dtype == "object"]

    # Contar valores únicos
    cardinality = {col: df[col].nunique() for col in object_cols}

    # Crear visualización
    plt.figure(figsize=(10, 6))
    bars = plt.bar(cardinality.keys(), cardinality.values(), color=sns.color_palette('viridis', len(cardinality)))

    # Añadir etiquetas y números
    plt.title('Cardinalidad de Variables Categóricas', fontsize=15)
    plt.xlabel('Columna', fontsize=12)
    plt.ylabel('Número de Valores Únicos', fontsize=12)
    plt.xticks(rotation=45)

    # Añadir los valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('cardinalidad_categoricas.png', dpi=300, bbox_inches='tight')
    plt.show()


def crear_visualizacion_conceptual():
    """
    Crea una visualización conceptual de la diferencia entre 
    codificación ordinal y one-hot.
    """
    # Datos de ejemplo
    original_data = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Color': ['Rojo', 'Verde', 'Azul', 'Rojo']
    })

    # Codificación Ordinal
    ordinal_data = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Color_Ordinal': [0, 1, 2, 0]  # Rojo=0, Verde=1, Azul=2
    })

    # Codificación One-Hot
    onehot_data = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Color_Rojo': [1, 0, 0, 1],
        'Color_Verde': [0, 1, 0, 0],
        'Color_Azul': [0, 0, 1, 0]
    })

    # Crear visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Datos originales
    axes[0].table(cellText=original_data.values, colLabels=original_data.columns,
                  loc='center', cellLoc='center')
    axes[0].set_title('Datos Originales', fontsize=12)
    axes[0].axis('off')

    # Codificación Ordinal
    axes[1].table(cellText=ordinal_data.values, colLabels=ordinal_data.columns,
                  loc='center', cellLoc='center')
    axes[1].set_title('Codificación Ordinal', fontsize=12)
    axes[1].axis('off')

    # Codificación One-Hot
    axes[2].table(cellText=onehot_data.values, colLabels=onehot_data.columns,
                  loc='center', cellLoc='center')
    axes[2].set_title('Codificación One-Hot', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Comparación de Métodos de Codificación Categórica', fontsize=16)
    plt.tight_layout()
    plt.savefig('categorical_encoding_viz.png', dpi=300, bbox_inches='tight')
    plt.show()


def identificar_columnas_seguras(X_train, X_valid):
    """
    Identifica columnas categóricas que pueden codificarse de forma segura 
    (todos los valores en validación están en entrenamiento).

    Retorna:
    - good_label_cols: Columnas seguras para codificación
    - bad_label_cols: Columnas con valores nuevos en validación
    """
    # Identificar columnas categóricas
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    # Columnas que pueden codificarse de forma segura
    good_label_cols = [col for col in object_cols if
                       set(X_valid[col]).issubset(set(X_train[col]))]

    # Columnas problemáticas que serán eliminadas del dataset
    bad_label_cols = list(set(object_cols) - set(good_label_cols))

    print(f"Columnas seguras para codificación: {good_label_cols}")
    print(f"Columnas con valores nuevos en validación: {bad_label_cols}")

    return good_label_cols, bad_label_cols


def analizar_cardinalidad(X_train, object_cols):
    """
    Analiza la cardinalidad (número de valores únicos) de cada columna categórica.
    """
    # Obtener número de valores únicos en cada columna categórica
    object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    # Imprimir número de valores únicos por columna, en orden ascendente
    cardinality_info = sorted(d.items(), key=lambda x: x[1])

    print("\nCardinalidad de columnas categóricas (ordenadas):")
    for col, count in cardinality_info:
        print(f"  {col}: {count} valores únicos")

    # Clasificar columnas por su cardinalidad
    low_cardinality = [col for col, count in d.items() if count <= 10]
    high_cardinality = [col for col, count in d.items() if count > 10]

    print(f"\nColumnas de baja cardinalidad (recomendadas para one-hot): {low_cardinality}")
    print(f"Columnas de alta cardinalidad (recomendadas para ordinal): {high_cardinality}")

    return low_cardinality, high_cardinality


def aplicar_one_hot(X_train, X_valid, columnas):
    """
    Aplica codificación one-hot a las columnas especificadas.
    """
    # Aplicar One-Hot Encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(X_train[columnas])

    # Transformar conjuntos de entrenamiento y validación
    OH_cols_train = pd.DataFrame(OH_encoder.transform(X_train[columnas]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[columnas]))

    # Una característica de la codificación one-hot es que pierde los nombres de columna
    # Podemos recuperarlos de las categorías del codificador
    OH_cols_train.columns = OH_encoder.get_feature_names_out(columnas)
    OH_cols_valid.columns = OH_encoder.get_feature_names_out(columnas)

    # Insertar índices para poder concatenar correctamente
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Quitar columnas originales y añadir las nuevas codificadas
    num_X_train = X_train.drop(columnas, axis=1)
    num_X_valid = X_valid.drop(columnas, axis=1)

    # Concatenar dataframes
    X_train_encoded = pd.concat([num_X_train, OH_cols_train], axis=1)
    X_valid_encoded = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    print(f"\nForma del dataset antes de one-hot: {X_train.shape}")
    print(f"Forma del dataset después de one-hot: {X_train_encoded.shape}")

    # Calcular cuántas entradas se han añadido
    added_entries = (X_train_encoded.shape[0] * X_train_encoded.shape[1]) - (X_train.shape[0] * X_train.shape[1])
    print(f"Entradas añadidas al dataset: {added_entries:,}")

    return X_train_encoded, X_valid_encoded


def aplicar_ordinal(X_train, X_valid, columnas):
    """
    Aplica codificación ordinal a las columnas especificadas.
    """
    # Aplicar Ordinal Encoding
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal_encoder.fit(X_train[columnas])

    # Crear copias para no modificar los originales
    X_train_encoded = X_train.copy()
    X_valid_encoded = X_valid.copy()

    # Aplicar transformación
    X_train_encoded[columnas] = ordinal_encoder.transform(X_train[columnas])
    X_valid_encoded[columnas] = ordinal_encoder.transform(X_valid[columnas])

    print(f"\nForma del dataset antes de ordinal encoding: {X_train.shape}")
    print(f"Forma del dataset después de ordinal encoding: {X_train_encoded.shape}")

    # Mostrar ejemplo de mapeo para la primera columna
    primera_columna = columnas[0]
    categorias = ordinal_encoder.categories_[0]
    print(f"\nEjemplo de mapeo ordinal para '{primera_columna}':")
    for i, categoria in enumerate(categorias):
        print(f"  {categoria} → {i}")

    return X_train_encoded, X_valid_encoded


def demostrar_impacto_memoria(n_rows=10000, n_categories=100):
    """
    Demuestra el impacto en memoria de aplicar one-hot encoding
    a una columna de alta cardinalidad.
    """
    print(f"\n{'=' * 50}")
    print("DEMOSTRACIÓN DE IMPACTO EN MEMORIA DE ONE-HOT ENCODING")
    print(f"{'=' * 50}")

    # Crear dataset sintético
    print(f"Creando dataset con {n_rows:,} filas y una columna categórica con {n_categories:,} valores únicos...")

    # Crear una columna categórica
    categorias = [f'cat_{i}' for i in range(n_categories)]
    data = pd.DataFrame({
        'categoria': np.random.choice(categorias, n_rows)
    })

    # Mostrar información antes de codificar
    print(
        f"Tamaño del dataset original: {data.shape[0]:,} filas × {data.shape[1]:,} columnas = {data.shape[0] * data.shape[1]:,} entradas")

    # Aplicar One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(data[['categoria']])

    # Convertir a DataFrame para mejor visualización
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=[f'categoria_{i}' for i in range(n_categories)]
    )

    # Mostrar información después de codificar
    print(
        f"Tamaño después de one-hot encoding: {encoded_df.shape[0]:,} filas × {encoded_df.shape[1]:,} columnas = {encoded_df.shape[0] * encoded_df.shape[1]:,} entradas")

    # Calcular entradas añadidas
    added_entries = (encoded_df.shape[0] * encoded_df.shape[1]) - (data.shape[0] * data.shape[1])
    print(f"Entradas añadidas: {added_entries:,}")

    # Visualizar el aumento
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = [data.shape[0] * data.shape[1], encoded_df.shape[0] * encoded_df.shape[1]]
    labels = ['Dataset Original', 'Dataset One-Hot Encoded']

    bars = ax.bar(labels, sizes, color=['#2ecc71', '#e74c3c'])

    # Formato de los números
    def format_num(x):
        if x >= 1_000_000:
            return f'{x / 1_000_000:.1f}M'
        elif x >= 1_000:
            return f'{x / 1_000:.1f}K'
        else:
            return str(x)

    # Añadir etiquetas
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                format_num(height), ha='center', fontsize=12)

    ax.set_title('Impacto de One-Hot Encoding en el Tamaño del Dataset', fontsize=15)
    ax.set_ylabel('Número de Entradas', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('impacto_one_hot.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Función principal que ejecuta la demostración completa.
    """
    print("\n" + "=" * 70)
    print(" DEMOSTRACIÓN DE CODIFICACIÓN DE VARIABLES CATEGÓRICAS ".center(70, "="))
    print("=" * 70 + "\n")

    # Crear datos de ejemplo
    print("Creando datos de ejemplo...")
    data = crear_datos_ejemplo()
    print(f"Conjunto de datos creado con {data.shape[0]} filas y {data.shape[1]} columnas.")
    print("\nPrimeras 5 filas del dataset:")
    print(data.head())

    # Dividir en entrenamiento y validación
    print("\nDividiendo datos en entrenamiento y validación...")
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} filas")
    print(f"Conjunto de validación: {X_valid.shape[0]} filas")

    # Identificar columnas seguras
    print("\n" + "-" * 70)
    print("IDENTIFICACIÓN DE COLUMNAS SEGURAS PARA CODIFICACIÓN".center(70, "-"))
    print("-" * 70)
    good_label_cols, bad_label_cols = identificar_columnas_seguras(X_train, X_valid)

    # Analizar cardinalidad
    print("\n" + "-" * 70)
    print("ANÁLISIS DE CARDINALIDAD".center(70, "-"))
    print("-" * 70)
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    low_cardinality, high_cardinality = analizar_cardinalidad(X_train, object_cols)

    # Visualizar cardinalidad
    print("\nCreando visualización de cardinalidad...")
    visualizar_cardinalidad(X_train)

    # Aplicar One-Hot Encoding a columnas de baja cardinalidad
    print("\n" + "-" * 70)
    print("APLICANDO ONE-HOT ENCODING A COLUMNAS DE BAJA CARDINALIDAD".center(70, "-"))
    print("-" * 70)
    if low_cardinality:
        X_train_oh, X_valid_oh = aplicar_one_hot(X_train, X_valid, low_cardinality)
    else:
        print("No se encontraron columnas de baja cardinalidad para one-hot encoding.")

    # Aplicar Ordinal Encoding a columnas de alta cardinalidad
    print("\n" + "-" * 70)
    print("APLICANDO ORDINAL ENCODING A COLUMNAS DE ALTA CARDINALIDAD".center(70, "-"))
    print("-" * 70)
    if high_cardinality:
        X_train_ord, X_valid_ord = aplicar_ordinal(X_train, X_valid, high_cardinality)
    else:
        print("No se encontraron columnas de alta cardinalidad para ordinal encoding.")

    # Crear visualización conceptual
    print("\nCreando visualización conceptual de métodos de codificación...")
    crear_visualizacion_conceptual()

    # Demostrar impacto en memoria
    demostrar_impacto_memoria(10000, 100)

    print("\n" + "=" * 70)
    print(" FIN DE LA DEMOSTRACIÓN ".center(70, "="))
    print("=" * 70)
    print("\nSe han guardado las visualizaciones en archivos PNG para su uso posterior.")


if __name__ == "__main__":
    main()