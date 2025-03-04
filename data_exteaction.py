import cv2
import numpy as np
import json
from scipy.signal import find_peaks
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Image Preprocessing
def preprocess_image(image_path):
    """
    Preprocess the ECG image: convert to grayscale, reduce noise, and remove grid lines.
    :param image_path: Path to the input ECG image
    :return: Preprocessed image
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Thresholding (for better edge detection and grid line removal)
    _, thresholded_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY_INV)

    # Remove grid lines (this can be adapted based on image properties)
    grid_removed_image = remove_grid_lines(thresholded_image)

    return grid_removed_image

def remove_grid_lines(image):
    """
    Detect and remove grid lines from the image.
    :param image: Binary image with ECG waveform
    :return: Image with grid lines removed
    """
    # Use morphological operations to remove lines
    kernel = np.ones((5, 5), np.uint8)
    image_dilated = cv2.dilate(image, kernel, iterations=1)
    grid_removed = cv2.erode(image_dilated, kernel, iterations=1)
    
    return grid_removed

# Lead Segmentation
def segment_leads(image, num_leads=12):
    """
    Automatically identify and extract individual leads from the ECG image.
    :param image: Preprocessed image with grid lines removed
    :param num_leads: Number of ECG leads to segment
    :return: List of segmented lead images
    """
    lead_height = image.shape[0]
    lead_width = image.shape[1] // num_leads
    leads = []
    
    for i in range(num_leads):
        lead_image = image[:, i*lead_width:(i+1)*lead_width]
        leads.append(lead_image)
    
    return leads

# Calibration
def calibrate_leads(leads, pixel_to_time=25, pixel_to_voltage=10):
    """
    Calibrate the leads by converting pixel values into real-world values (time and voltage).
    :param leads: List of segmented lead images
    :param pixel_to_time: Pixel to time conversion factor (mm/s)
    :param pixel_to_voltage: Pixel to voltage conversion factor (mm/mV)
    :return: List of calibrated leads
    """
    calibrated_leads = []
    for lead in leads:
        # Convert pixel values to time and voltage (simplified approach)
        time = np.linspace(0, len(lead) / pixel_to_time, len(lead))
        voltage = np.array([np.mean(lead[i:i+10]) for i in range(0, len(lead), 10)]) * pixel_to_voltage
        calibrated_leads.append((time, voltage))
    
    return calibrated_leads

# Waveform Extraction (Modified to give time difference and amplitude)
def extract_waveform_data(lead_data):
    """
    Extract key parameters from each lead, including P wave, QRS complex, and T wave.
    :param lead_data: Tuple of time and voltage data for the lead
    :return: Dictionary with extracted wave parameters (P, QRS, T) with time difference and amplitude
    """
    time, voltage = lead_data

    # Detect P wave, QRS complex, and T wave peaks using find_peaks
    p_peaks, _ = find_peaks(voltage, height=0.5, distance=30)
    qrs_peaks, _ = find_peaks(voltage, height=1.0, distance=10)
    t_peaks, _ = find_peaks(voltage, height=0.5, distance=50)

    # Extract metrics for each wave (time difference and amplitude)
    waves = {
        "P_wave": extract_wave_metrics(time, voltage, p_peaks),
        "QRS_complex": extract_qrs_metrics(time, voltage, qrs_peaks),
        "T_wave": extract_wave_metrics(time, voltage, t_peaks),
    }

    return waves

def extract_wave_metrics(time, voltage, peaks):
    """
    Extract the time difference and amplitude for a given wave.
    :param time: Time array for the ECG lead
    :param voltage: Voltage array for the ECG lead
    :param peaks: List of indices where peaks occur
    :return: Dictionary with time_difference and amplitude for the wave
    """
    wave_metrics = []
    for peak in peaks:
        # Estimate start before peak and end after peak (based on peak position)
        start = time[max(0, peak-10)]  
        end = time[min(len(time)-1, peak+10)]  
        amplitude = voltage[peak]  # Peak amplitude
        time_difference = end - start  # Calculate time difference
        wave_metrics.append({
            "amplitude": amplitude,
            "time_difference": time_difference  # Time difference between start and end
        })
    return wave_metrics

def extract_qrs_metrics(time, voltage, qrs_peaks):
    """
    Extract time difference and amplitude for individual Q, R, and S components of the QRS complex.
    :param time: Time array for the ECG lead
    :param voltage: Voltage array for the ECG lead
    :param qrs_peaks: Indices where QRS peaks occur
    :return: Dictionary with time difference and amplitude data for Q, R, S waves
    """
    qrs_metrics = {
        "Q": extract_wave_metrics(time, voltage, qrs_peaks),
        "R": extract_wave_metrics(time, voltage, qrs_peaks),
        "S": extract_wave_metrics(time, voltage, qrs_peaks)
    }
    return qrs_metrics

# Output Formatting (For just second lead)
def generate_json_output_for_second_lead(calibrated_leads):
    """
    Generate the final JSON output for just the second lead with time difference and amplitude.
    :param calibrated_leads: List of calibrated lead data
    :return: JSON formatted string with waveform data for the second lead
    """
    second_lead_data = calibrated_leads[1]  # Extract the second lead (index 1)
    waves = extract_waveform_data(second_lead_data)

    # Convert to JSON format
    json_output = json.dumps(waves, indent=4)
    return json_output

def main(image_path):
    """
    Main function to process ECG image and output waveform data in JSON format for the second lead.
    :param image_path: Path to the ECG image file
    """
    try:
        # Step 1: Preprocess the image
        logging.info("Preprocessing the ECG image...")
        preprocessed_image = preprocess_image(image_path)

        # Step 2: Segment leads
        logging.info("Segmenting the ECG leads...")
        leads = segment_leads(preprocessed_image)

        # Step 3: Calibrate leads (convert pixels to time and voltage)
        logging.info("Calibrating leads...")
        calibrated_leads = calibrate_leads(leads)

        # Step 4: Generate output for the second lead
        logging.info("Generating JSON output for second lead...")
        json_output = generate_json_output_for_second_lead(calibrated_leads)

        # Print or save the output
        with open('ecg_waveform_data_second_lead.json', 'w') as json_file:
            json_file.write(json_output)
        logging.info("ECG waveform data for the second lead has been successfully written to 'ecg_waveform_data_second_lead.json'.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    image_path = r'C:\Users\User\Downloads\test\ECG Images of Myocardial Infarction Patients (240x12=2880)\MI(1).jpg'  # Replace with actual image path
    main(image_path)
