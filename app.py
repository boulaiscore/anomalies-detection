from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

app = Flask(__name__)

# Crea una cartella per caricare i file temporaneamente
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Endpoint per caricare un file Excel
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200

# Endpoint per ottenere i nomi delle colonne dal file Excel caricato
@app.route('/get_columns', methods=['POST'])
def get_columns():
    data = request.json
    file_path = data.get("file_path")
    
    if not file_path:
        return jsonify({"error": "No file path provided"}), 400

    # Carica il file ed estrae i nomi delle colonne
    df = pd.read_excel(file_path)
    columns = df.columns.tolist()
    
    return jsonify(columns), 200

# Funzione di anomaly detection con Prophet
def detect_anomalies_with_prophet(file_path, metric_col='y', date_col='ds'):
    # Carica i dati dal file Excel
    df = pd.read_excel(file_path)
    data = df[[date_col, metric_col]].rename(columns={date_col: 'ds', metric_col: 'y'})

    # Converte la colonna 'ds' in formato datetime
    data['ds'] = pd.to_datetime(data['ds'])

    # Inizializza il modello Prophet
    model = Prophet(interval_width=0.95)  # Intervallo di confidenza al 95%
    model.fit(data)
    
    # Prepara le previsioni future
    future = model.make_future_dataframe(periods=30, freq='D')  # Previsione per i prossimi 30 giorni
    forecast = model.predict(future)
    
    # Unisce le previsioni con i valori reali
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    merged = pd.merge(data, forecast, on='ds')
    
    # Rileva anomalie
    merged['Anomaly'] = (merged['y'] > merged['yhat_upper']) | (merged['y'] < merged['yhat_lower'])
    
    # Filtra solo le anomalie
    anomalies = merged[merged['Anomaly'] == True]
    
    # Genera il grafico
    plot_anomalies(merged, anomalies)

    return anomalies

# Funzione per visualizzare anomalie con una migliore rappresentazione grafica
def plot_anomalies(forecast, anomalies):
    plt.figure(figsize=(12, 6))

    # Linee più sottili e punti per i dati osservati
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='green', linewidth=1.5)
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightgrey', alpha=0.5, label='Confidence Interval')
    
    # Punti per i dati osservati
    plt.scatter(forecast['ds'], forecast['y'], label='Observed', color='blue', alpha=0.7, s=10)
    
    # Evidenziamo le anomalie con cerchi più grandi
    plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies', s=80, edgecolor='black', linewidth=1.5)

    plt.xlabel('Date')
    plt.ylabel('Metric Value')
    plt.title('Anomalies Detection in Time Series')

    # Formattazione dell'asse delle date per renderlo più leggibile
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()

    plt.legend()
    plt.grid(True)
    plt.show()

# Endpoint per l'analisi delle anomalie
@app.route('/detect_time_series_anomalies', methods=['POST'])
def detect_time_series_anomalies():
    data = request.json
    file_path = data.get("file_path")
    metric_col = data.get("metric_col", "y")  # Metrica da analizzare
    date_col = data.get("date_col", "ds")  # Colonna data
    
    if not file_path:
        return jsonify({"error": "No file path provided"}), 400

    # Rilevamento anomalie
    anomalies = detect_anomalies_with_prophet(file_path, metric_col=metric_col, date_col=date_col)
    return anomalies.to_json(orient='records'), 200

# Avvio del server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
