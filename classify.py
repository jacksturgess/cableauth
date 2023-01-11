#python classify.py [basedir] [userID(s)] [param(s)] [device(s)] [mode(s)] [poollimit] [is_restrictedsensors] [number_of_trees]
# - basedir
# - userID(s) (may be a list)
# - param(s): f<windowsize>o<offset> (each is a list of {CAR_out_param,CHARGER_in_param,CHARGER_out_param,CAR_in_param}; may be a "&"-SEPARATED list)
# - device(s): a (all) or list of any of {watch, ring}
# - mode(s): auth (may be a list)
# - poollimit: integer (number of parallel threads to use)
# - is_restrictedsensors: true or false (for whether to use only acc/gyr sensors) (may be a list)
# - number_of_trees: integer (number of trees in the random forests)
#
#opens the file <userdir>/1-cleaned/gestures-<device>.csv, grabs data according to <param>, applies a low pass filter to each gesture, extracts features, saves these feature vectors to file, and then classifies
#outputs <userdir>/2-processed/<device>-<gesture_type>-<param>-features.csv and <datetime>-<userIDs>-<device>-<param>-<mode>.csv

import datetime, csv, math, os, re, shutil, statistics, sys
import numpy as np
from multiprocessing import Pool
from scipy.signal import butter, lfilter, freqz, find_peaks
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

basedir = sys.argv[1] + '/'
userIDs = re.split(',', (sys.argv[2]).lower())
params = re.split('&', (sys.argv[3]).lower())
devices = ['watch', 'ring'] if 'a' == (sys.argv[4]).lower() else re.split(',', sys.argv[4])
modes = re.split(',', (sys.argv[5]).lower())
poollimit = int(sys.argv[6])
is_restrictedsensorses = re.split(',', sys.argv[7])
number_of_trees = int(sys.argv[8])

#configs
maxpoolsize = 36
maxprewindowsize = 4
minsamplespergesture = 80
repetitions = 10

#consts
g_index = 0 #gesture name and index
s_index = 1 #sensor name
t_index = 2 #gesture timestamp (in seconds, range from prewindowsize to 0)
x_index = 3 #x-value
y_index = 4 #y-value
z_index = 5 #z-value
w_index = 6 #w-value (GRV only)
e_unf_index = 7 #Euclidean norm of unfiltered values
e_fil_index = 8 #Euclidean norm of filtered values

#values
f_names = [] #container to hold the names of the features
sensors = [] #container to hold the sensors used by the dataset
data_acc = []
data_gyr = []
data_grv = []
data_lac = []

def get_tidy_userIDs():
	global userIDs
	t_userIDs = []
	for userID in userIDs:	
		if 'user' in userID:
			t_userIDs.append('user' + f'{int(userID[4:]):03}')
		else:
			t_userIDs.append('user' + f'{int(userID):03}')
	t_userIDs.sort(reverse = False)
	return(t_userIDs)

def get_tidy_params():
	global params
	t_params = []
	for param in params:
		s_params = []
		subparams = re.split(',', param)
		for subparam in subparams:
			windowsize = 0
			offset = 0
			if 'o' in subparam:
				if 'om' in subparam:
					subparam = re.sub('om', 'o-', subparam)
				windowsize = float(subparam[1:subparam.index('o')])
				offset = float(subparam[subparam.index('o') + 1:])
			else:
				windowsize = float(subparam[1:])
			if windowsize + offset > maxprewindowsize:
				windowsize = maxprewindowsize - offset
			windowsize = str('%.1f' % windowsize)
			offset = str('%.1f' % offset)
			if 'f' == subparam[0]:
				s_params.append('f' + windowsize + 'o' + offset)
			elif 'u' == subparam[0]:
				s_params.append('u' + windowsize + 'o' + offset)
			else:
				sys.exit('ERROR: subparam not valid: ' + subparam)
		t_params.append(s_params)
	return(t_params)

def get_gesture_type(i):
	if 0 == i:
		return 'CAR_out'
	elif 1 == i:
		return 'CHARGER_in'
	elif 2 == i:
		return 'CHARGER_out'
	elif 3 == i:
		return 'CAR_in'

def get_features(data, is_restrictedsensors):
	featurecolumns = []
	f_names = data[1:]
	f_column = 1
	sensor_list = ['Acc', 'Gyr'] if is_restrictedsensors else ['Acc', 'Gyr', 'GRV', 'LAc']
	for f_name in f_names:
		s = re.split('-', f_name)[0]
		for sensor in sensor_list:
			if s == sensor:
				featurecolumns.append(f_column)
				break;
		f_column = f_column + 1
	featurecolumns.sort(reverse = False)
	featurenames = [f_names[c - 1] for c in featurecolumns]
	return featurenames, featurecolumns

def get_average(l):
	return 0 if 0 == len(l) else sum(l) / len(l)

def get_eer(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	#treat each legitimate sample distance as a possible threshold, determine the point where FRR crosses FAR
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr >= far:
			return threshold, far
	return 1, 1

def get_far_when_zero_frr(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	#treat each legitimate sample distance as a possible threshold, determine the point with the lowest FAR that satisfies the condition that FRR = 0
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr > 0.001:
			return threshold, far
	return 1, 1

def get_frr_when_zero_far(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit, reverse = True)
	scores_adv = sorted(scores_adv, reverse = True)
	
	#treat each legitimate sample distance as a possible threshold, determine the point with the lowest FRR that satisfies the condition that FAR = 0
	for c, threshold in enumerate(scores_adv):
		far = c * 1.0 / len(scores_adv)
		legit_index = next((x[0] for x in enumerate(scores_legit) if x[1] < threshold), len(scores_legit))
		if len(scores_legit) > 0:
			frr = 1 - (legit_index * 1.0 / len(scores_legit))
			if far > 0.001:
				return threshold, frr
	return 1, 1

def get_ascending_userID_list_string(userIDs):
	for u in userIDs:
		if not 'user' in u and len(u) != 7:
			sys.exit('ERROR: userID not valid: ' + str(u))
	IDs = [int(u[4:]) for u in userIDs]
	IDs.sort(reverse = False)
	return ','.join([f'{i:03}' for i in IDs])

def get_descending_feature_list_string(weights, labels, truncate = 0):
	indicies = [i for i in range(len(weights))]
	for i in range(len(indicies)):
		for j in range(len(indicies)):
			if i != j and weights[indicies[i]] > weights[indicies[j]]:
				temp = indicies[i]
				indicies[i] = indicies[j]
				indicies[j] = temp
	if truncate != 0:
		del indicies[truncate:]
	return '\n'.join([str('%.6f' % weights[i]) + ' (' + labels[i] + ')' for i in indicies])

def write_verbose(f, s):
	f_outfilename = f + '-verbose.txt'
	outfile = open(f_outfilename, 'a')
	outfile.write(s + '\n')
	outfile.close()

def rewrite_param(param):
	t_param = param
	if 'o-' in t_param:
		t_param = re.sub('o-', 'om', t_param)
	return t_param

def butter_lowpass(cutoff, r, order):
    nyq = 0.5 * r
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    return (b, a)

def butter_lowpass_filter(d, cutoff, r, order):
    b, a = butter_lowpass(cutoff, r, order)
    y = lfilter(b, a, d)
    return y

def filterGesture(data, name, device):
	output = []
	sensors = ['Acc', 'Gyr', 'GRV', 'LAc']
	for s in sensors:
		times = []
		x = []
		y = []
		z = []
		w = []
		e = []
		for datum in data:
			if s == datum[s_index]:
				times.append(float(datum[3]))
				x.append(float(datum[4]))
				y.append(float(datum[5]))
				z.append(float(datum[6]))
				if 'GRV' == s:
					w.append(float(datum[7]))
				else:
					e.append(datum[8])
		
		#filtering variables
		order = 6
		cutoff = 3.667 #desired cutoff frequency of the filter, Hz
		n = len(times) #total number of samples
		t = np.linspace(times[0], times[len(times) - 1], n, endpoint = False) #evenly spaced time intervals
		r = int(n / (times[len(times) - 1] - times[0])) #sample rate, Hz
		
		b, a = butter_lowpass(cutoff, r, order) #gets the filter coefficients so we can check its frequency response
		filtered_x = butter_lowpass_filter(x, cutoff, r, order)
		filtered_y = butter_lowpass_filter(y, cutoff, r, order)
		filtered_z = butter_lowpass_filter(z, cutoff, r, order)
		filtered_w = []
		if 'GRV' == s:
			filtered_w = butter_lowpass_filter(w, cutoff, r, order)
		
		#restructure data for output
		for i in range(len(t)):
			d_x = filtered_x[i]
			d_y = filtered_y[i]
			d_z = filtered_z[i]
			datum = []
			datum.append(name) #adds gesture name and index
			datum.append(s) #adds sensor name
			datum.append(str('%.6f' % t[i])) #adds new timestamp
			datum.append(str('%.6f' % d_x)) #adds filtered x-value
			datum.append(str('%.6f' % d_y)) #adds filtered y-value
			datum.append(str('%.6f' % d_z)) #adds filtered z-value
			if 'GRV' == s:
				datum.append(str('%.6f' % filtered_w[i])) #adds filtered w-value
				datum.append('') #adds empty field instead of Euclidean norm of unfiltered values
				datum.append('') #adds empty field instead of Euclidean norm of filtered values
			else:
				datum.append('') #adds empty field instead of filtered w-value
				datum.append(e[i]) #adds Euclidean norm of unfiltered values
				datum.append(str('%.6f' % math.sqrt(d_x * d_x + d_y * d_y + d_z * d_z))) #adds Euclidean norm of filtered values
			output.append(datum)			
	return output

def write_f_name(s):
	if 'Acc' in sensors:
		for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
			f_names.append('Acc-' + dimension + s)
	if 'Gyr' in sensors:
		for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
			f_names.append('Gyr-' + dimension + s)
	if 'GRV' in sensors:
		for dimension in ['x-', 'y-', 'z-', 'w-']:
			f_names.append('GRV-' + dimension + s)
	if 'LAc' in sensors:
		for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
			f_names.append('LAc-' + dimension + s)

def feature_min(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('min')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % min(datum)))
	return ','.join(f)

def feature_max(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('max')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % max(datum)))
	return ','.join(f)

def feature_mean(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('mean')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % statistics.mean(datum)))
	return ','.join(f)

def feature_med(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('med')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % statistics.median(datum)))
	return ','.join(f)
	
def feature_stdev(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('stdev')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % np.std(datum, ddof = 1)))
	return ','.join(f)

def feature_var(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('var')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % np.var(datum, ddof = 1)))
	return ','.join(f)

def feature_iqr(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('iqr')
	
	f = []
	for datum in g_data:
		q75, q25 = np.percentile(datum, [75, 25])
		f.append(str('%.6f' % (q75 - q25)))
	return ','.join(f)

def feature_kurt(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('kurt')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % kurtosis(datum)))
	return ','.join(f)

def feature_skew(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('skew')
	f = []
	for datum in g_data:
		f.append(str('%.6f' % skew(datum)))
	return ','.join(f)

def feature_pkcount(is_firstparse, g_data, threshold):
	if is_firstparse:
		write_f_name('pkcount' + str(threshold))
	f = []
	for datum in g_data:
		f.append(str('%.0f' % len(find_peaks(datum, prominence = threshold)[0])))
	return ','.join(f)

def feature_velo_disp(is_firstparse):
	f = []
	for sensor in ['Acc', 'Gyr', 'LAc']:
		if sensor in sensors:
			s_data = []
			if 'Acc' == sensor:
				s_data = list(data_acc)
			elif 'Gyr' == sensor:
				s_data = list(data_gyr)
			elif 'LAc' == sensor:
				s_data = list(data_lac)
			
			vx = [0]
			dx = [0]
			vy = [0]
			dy = [0]
			vz = [0]
			dz = [0]
			d = [0]
			n = len(s_data) - 1 #number of samples
			dt = float((s_data[n][0] - s_data[0][0]) / n) #sample interval
			for j in range(n):
				vx.append(vx[j] + (s_data[j][1] + s_data[j + 1][1]) / 2 * dt / 10)
				dx.append(dx[j] + vx[j + 1] * dt / 10)
				vy.append(vy[j] + (s_data[j][2] + s_data[j + 1][2]) / 2 * dt / 10)
				dy.append(dy[j] + vy[j + 1] * dt / 10)
				vz.append(vz[j] + (s_data[j][3] + s_data[j + 1][3]) / 2 * dt / 10)
				dz.append(dz[j] + vz[j + 1] * dt / 10)
				d.append(math.sqrt(dx[j] * dx[j] + dy[j] * dy[j] + dz[j] * dz[j]))
			vx.pop(0)
			vy.pop(0)
			vz.pop(0)
			
			if is_firstparse:
				f_names.append(sensor + '-x-velomean')
				f_names.append(sensor + '-y-velomean')
				f_names.append(sensor + '-z-velomean')
				f_names.append(sensor + '-x-velomax')
				f_names.append(sensor + '-y-velomax')
				f_names.append(sensor + '-z-velomax')
				f_names.append(sensor + '-x-disp')
				f_names.append(sensor + '-y-disp')
				f_names.append(sensor + '-z-disp')
				f_names.append(sensor + '-disptotal')
			
			f.append(str('%.6f' % (sum(vx) / len(vx))))
			f.append(str('%.6f' % (sum(vy) / len(vy))))
			f.append(str('%.6f' % (sum(vz) / len(vz))))
			f.append(str('%.6f' % max(vx, key = abs)))
			f.append(str('%.6f' % max(vy, key = abs)))
			f.append(str('%.6f' % max(vz, key = abs)))
			f.append(str('%.6f' % dx[len(dx) - 1]))
			f.append(str('%.6f' % dy[len(dy) - 1]))
			f.append(str('%.6f' % dz[len(dz) - 1]))
			f.append(str('%.6f' % d[len(d) - 1]))
	return ','.join(f)

def extractFeatures(is_firstparse):
	#prepare data
	g_data = []
	if 'Acc' in sensors:
		g_data.append([row[1] for row in data_acc])
		g_data.append([row[2] for row in data_acc])
		g_data.append([row[3] for row in data_acc])
		g_data.append([row[4] for row in data_acc])
		g_data.append([row[5] for row in data_acc])
	if 'Gyr' in sensors:
		g_data.append([row[1] for row in data_gyr])
		g_data.append([row[2] for row in data_gyr])
		g_data.append([row[3] for row in data_gyr])
		g_data.append([row[4] for row in data_gyr])
		g_data.append([row[5] for row in data_gyr])
	if 'GRV' in sensors:
		g_data.append([row[1] for row in data_grv])
		g_data.append([row[2] for row in data_grv])
		g_data.append([row[3] for row in data_grv])
		g_data.append([row[4] for row in data_grv])
	if 'LAc' in sensors:
		g_data.append([row[1] for row in data_lac])
		g_data.append([row[2] for row in data_lac])
		g_data.append([row[3] for row in data_lac])
		g_data.append([row[4] for row in data_lac])
		g_data.append([row[5] for row in data_lac])
	
	#call features for this gesture
	f_data = []
	f_data.append(feature_min(is_firstparse, g_data))
	f_data.append(feature_max(is_firstparse, g_data))
	f_data.append(feature_mean(is_firstparse, g_data))
	f_data.append(feature_med(is_firstparse, g_data))
	f_data.append(feature_stdev(is_firstparse, g_data))
	f_data.append(feature_var(is_firstparse, g_data))
	f_data.append(feature_iqr(is_firstparse, g_data))
	f_data.append(feature_kurt(is_firstparse, g_data))
	f_data.append(feature_skew(is_firstparse, g_data))
	f_data.append(feature_pkcount(is_firstparse, g_data, 0.5))
	f_data.append(feature_velo_disp(is_firstparse))
	return ','.join(f_data)

def process(args):
	basedir, userIDs, device, mode, param, is_restrictedsensors = args
	
	for userID in userIDs:
		sourcedir = basedir + userID + '/0-raw/'
		if not os.path.exists(sourcedir):
			sys.exit('ERROR: no such sourcedir for user: ' + userID)
		
		cleandir = basedir + userID + '/1-cleaned/'	
		if not os.path.exists(cleandir):
			sys.exit('ERROR: no such cleandir for user: ' + userID)
		
		extractdir = basedir + userID + '/2-extracted/'	
		
		f_infile = cleandir + 'gestures-' + device + '.csv'
		if os.path.exists(f_infile):
			with open(f_infile, 'r') as f:
				data = list(csv.reader(f)) #returns a list of lists (each line is a list)
				data.pop(0) #removes the column headers
				
				#build list of gesture indices
				gestureindices = []
				for datum in data:
					if datum[g_index] not in gestureindices:
						gestureindices.append(datum[g_index])
				
				for i in range(4):
					f_outfilename = extractdir + device + '-' + get_gesture_type(i) + '-' + rewrite_param(param[i]) + '-features.csv'
					if not os.path.exists(f_outfilename):
						windowsize = str('%.1f' % float(param[i][1:(param[i]).index('o')]))
						if '0.0' != windowsize:
							offset = str('%.1f' % float(param[i][(param[i]).index('o') + 1:]))
							endtime = -float(offset)
							starttime = endtime - float(windowsize)
							
							is_firstparse = True
							f_names.clear()
							sensors.clear()
							
							#for each gesture index: project a time window (backwards from the end), grab the data inside that window, apply a low pass filter to it, and write it
							for gestureindex in gestureindices:
								if get_gesture_type(i) in gestureindex:
									g_output = []
									for datum in data:
										if datum[g_index] == gestureindex:
											t = float(datum[3])
											if t >= starttime and t <= endtime:
												g_output.append(datum)
									
									#only process gesture if it has sufficient data (relevant for small time windows) for all devices (relevant for fair comparison of performance of devices)
									if len(g_output) >= minsamplespergesture:
										#filter and restructure data
										g_output = filterGesture(g_output, gestureindex, device)
										
										data_acc.clear()
										data_gyr.clear()
										data_grv.clear()
										data_lac.clear()
										
										#extract features
										for datum in g_output:
											s = datum[s_index]
											d = []
											d.append(float(datum[t_index]))
											d.append(float(datum[x_index]))
											d.append(float(datum[y_index]))
											d.append(float(datum[z_index]))
											if 'GRV' == s:
												d.append(float(datum[w_index]))
											else:
												d.append(float(datum[e_unf_index]))
												d.append(float(datum[e_fil_index]))
											
											if 'Acc' == s:
												data_acc.append(d)	
											elif 'Gyr' == s:
												data_gyr.append(d)
											elif 'GRV' == s:
												data_grv.append(d)
											elif 'LAc' == s:
												data_lac.append(d)
											
											if len(data_acc) > 0:
												sensors.append('Acc')
											if len(data_gyr) > 0:
												sensors.append('Gyr')
											if len(data_grv) > 0:
												sensors.append('GRV')
											if len(data_lac) > 0:
												sensors.append('LAc')
										
										#extract features
										f_output = extractFeatures(is_firstparse)
										
										#output features to the combined file
										f_outfile = open(f_outfilename, 'a')
										if is_firstparse:
											f_outfile.write('GESTURE,' + ','.join(f_names))
											is_firstparse = False
										f_outfile.write('\n' + gestureindex + ',' + f_output)
										f_outfile.close()
									else:
										print(' Rejected: ' + gestureindex + ' (' + str(len(g_output)) + ' samples)')
							print('OUTPUT: ' + f_outfilename)
	
	#classify
	print('CLASSIFY: ' + device + ', ' + '-'.join(param) + ', ' + mode)
	
	param_string = '-'.join([rewrite_param(param[i]) for i in range(4)])
	sensor_string = '-acc,gyr' if is_restrictedsensors else ''
	filename_string = datetime.datetime.now().strftime('%Y%m%d') + '-' + get_ascending_userID_list_string(userIDs) + '-' + device + '-' + param_string + '-' + mode + sensor_string
	f_outfilename = filename_string + '.csv'
	
	output = []
	
	a_data = [[], [], [], []] #container to hold the feature data for all users
	a_labels = [[], [], [], []] #container to hold the corresponding labels
	a_precisions = []
	a_recalls = []
	a_fmeasures = []
	a_pr_stdev = []
	a_re_stdev = []
	a_fm_stdev = []
	a_eers = []
	a_eer_thetas = []
	a_fars = []
	a_far_thetas = []
	a_frrs = []
	a_frr_thetas = []
	a_ee_stdev = []
	a_ee_th_stdev = []
	a_fa_stdev = []
	a_fa_th_stdev = []
	a_fr_stdev = []
	a_fr_th_stdev = []
	featurenames = [] #container to hold the names of the features
	
	if 'auth' == mode:
		output.append('userID,prec_avg,prec_stdev,rec_avg,rec_stdev,fm_avg,fm_stdev,eer_avg,eer_stdev,eer_theta_avg,eer_theta_stdev,far_avg,far_stdev,far_theta_avg,far_theta_stdev,frr_avg,frr_stdev,frr_theta_avg,frr_theta_stdev')
		
		#get feature data and labels for all users
		for userID in userIDs:
			extractdir = basedir + userID + '/2-extracted/'	
			
			for i in range(4):
				windowsize = str('%.1f' % float(param[i][1:(param[i]).index('o')]))
				if '0.0' != windowsize:
					gesture_type = get_gesture_type(i)
					f_infile = extractdir + device + '-' + gesture_type + '-' + rewrite_param(param[i]) + '-features.csv'
					if os.path.exists(f_infile):
						with open(f_infile, 'r') as f:
							data = list(csv.reader(f)) #returns a list of lists (each line is a list)
							featurenames, featurecolumns = get_features([h + '-' + gesture_type for h in data[0]], is_restrictedsensors)
							data.pop(0) #removes the column headers
							
							for datum in data:
								d = [datum[0]]
								d.extend([float(datum[n]) for n in featurecolumns])
								a_data[i].append(d)
								a_labels[i].append(userID)
		
		#run tests
		for userID in userIDs:
			u_precisions = []
			u_recalls = []
			u_fmeasures = []
			u_eers = []
			u_eer_thetas = []
			u_fars = []
			u_far_thetas = []
			u_frrs = []
			u_frr_thetas = []
			
			data_train = []
			data_test = []
			labels_train = []
			labels_test = []
			
			for i in range(4):
				if len(a_labels[i]) > 0:
					labels_train_i = []
					for j in range(len(a_data[i])):
						u = a_labels[i][j]
						if labels_train_i.count(u) < int(a_labels[i].count(u) * 2 / 3) + 1:
							data_train.append(a_data[i][j][1:])
							labels_train.append(1 if userID == u else 0)
							labels_train_i.append(u)
						else:
							data_test.append(a_data[i][j][1:])
							labels_test.append(1 if userID == u else 0)
			
			for repetition in range(repetitions):
				model = RandomForestClassifier(n_estimators = number_of_trees, random_state = repetition).fit(data_train, labels_train)
				labels_pred = model.predict(data_test)
				
				#get precision, recall, and F-measure scores
				precision = precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				recall = recall_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				fmeasure = f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				u_precisions.append(precision)
				u_recalls.append(recall)
				u_fmeasures.append(fmeasure)
				
				#get EER and find the decision threshold and FAR when optimised for FRR
				labels_scores = model.predict_proba(data_test)[:, 1]
				scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
				scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
				eer_theta, eer = get_eer(scores_legit, scores_adv)
				u_eers.append(eer)
				u_eer_thetas.append(eer_theta)
				far_theta, far = get_far_when_zero_frr(scores_legit, scores_adv)
				u_fars.append(far)
				u_far_thetas.append(far_theta)
				frr_theta, frr = get_frr_when_zero_far(scores_legit, scores_adv)
				u_frrs.append(frr)
				u_frr_thetas.append(frr_theta)
				
				write_verbose(filename_string, '----\n----USERID ' + userID + ', REPETITION ' + str(repetition) +
				 '\n----\nVALUES: precision=' + str('%.6f' % precision) + ', recall=' + str('%.6f' % recall) + ', fmeasure=' + str('%.6f' % fmeasure) +
				 ', eer=' + str('%.6f' % eer) + ', eer_theta=' + str('%.6f' % eer_theta) + ', far=' + str('%.6f' % far) + ', far_theta=' + str('%.6f' % far_theta) + ', frr=' + str('%.6f' % frr) + ', frr_theta=' + str('%.6f' % frr_theta) +
				 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
				 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
			u_pr_stdev = np.std(u_precisions, ddof = 1)
			u_re_stdev = np.std(u_recalls, ddof = 1)
			u_fm_stdev = np.std(u_fmeasures, ddof = 1)
			u_ee_stdev = np.std(u_eers, ddof = 1)
			u_ee_th_stdev = np.std(u_eer_thetas, ddof = 1)
			u_fa_stdev = np.std(u_fars, ddof = 1)
			u_fa_th_stdev = np.std(u_far_thetas, ddof = 1)
			u_fr_stdev = np.std(u_frrs, ddof = 1)
			u_fr_th_stdev = np.std(u_frr_thetas, ddof = 1)
			
			result_string = (userID + ',' + str('%.6f' % get_average(u_precisions)) + ',' + str('%.6f' % u_pr_stdev) + ','
			 + str('%.6f' % get_average(u_recalls)) + ',' + str('%.6f' % u_re_stdev) + ','
			 + str('%.6f' % get_average(u_fmeasures)) + ',' + str('%.6f' % u_fm_stdev) + ','
			 + str('%.6f' % get_average(u_eers)) + ',' + str('%.6f' % u_ee_stdev) + ','
			 + str('%.6f' % get_average(u_eer_thetas)) + ',' + str('%.6f' % u_ee_th_stdev) + ','
			 + str('%.6f' % get_average(u_fars)) + ',' + str('%.6f' % u_fa_stdev) + ','
			 + str('%.6f' % get_average(u_far_thetas)) + ',' + str('%.6f' % u_fa_th_stdev) + ','
			 + str('%.6f' % get_average(u_frrs)) + ',' + str('%.6f' % u_fr_stdev) + ','
			 + str('%.6f' % get_average(u_frr_thetas)) + ',' + str('%.6f' % u_fr_th_stdev)
			 )
			output.append(result_string)
			#print(result_string)
			
			a_precisions.extend(u_precisions)
			a_recalls.extend(u_recalls)
			a_fmeasures.extend(u_fmeasures)
			a_pr_stdev.append(u_pr_stdev)
			a_re_stdev.append(u_re_stdev)
			a_fm_stdev.append(u_fm_stdev)
			a_eers.extend(u_eers)
			a_eer_thetas.extend(u_eer_thetas)
			a_fars.extend(u_fars)
			a_far_thetas.extend(u_far_thetas)
			a_frrs.extend(u_frrs)
			a_frr_thetas.extend(u_frr_thetas)
			a_ee_stdev.append(u_ee_stdev)
			a_ee_th_stdev.append(u_ee_th_stdev)
			a_fa_stdev.append(u_fa_stdev)
			a_fa_th_stdev.append(u_fa_th_stdev)
			a_fr_stdev.append(u_fr_stdev)
			a_fr_th_stdev.append(u_fr_th_stdev)
		result_string = ('average,' + str('%.6f' % get_average(a_precisions)) + ',' + str('%.6f' % get_average(a_pr_stdev)) + ','
		 + str('%.6f' % get_average(a_recalls)) + ',' + str('%.6f' % get_average(a_re_stdev)) + ','
		 + str('%.6f' % get_average(a_fmeasures)) + ',' + str('%.6f' % get_average(a_fm_stdev)) + ','
		 + str('%.6f' % get_average(a_eers)) + ',' + str('%.6f' % get_average(a_ee_stdev)) + ','
		 + str('%.6f' % get_average(a_eer_thetas)) + ',' + str('%.6f' % get_average(a_ee_th_stdev)) + ','
		 + str('%.6f' % get_average(a_fars)) + ',' + str('%.6f' % get_average(a_fa_stdev)) + ','
		 + str('%.6f' % get_average(a_far_thetas)) + ',' + str('%.6f' % get_average(a_fa_th_stdev)) + ','
		 + str('%.6f' % get_average(a_frrs)) + ',' + str('%.6f' % get_average(a_fr_stdev)) + ','
		 + str('%.6f' % get_average(a_frr_thetas)) + ',' + str('%.6f' % get_average(a_fr_th_stdev))
		 )
		output.append(result_string)
	else:
		sys.exit('ERROR: mode not valid: ' + mode)
	
	outfile = open(f_outfilename, 'w')
	outfile.write('\n'.join(output))
	outfile.close()
	print('OUTPUT: ' + f_outfilename)

if __name__ == '__main__':
	userIDs = get_tidy_userIDs()
	params = get_tidy_params()
	
	if not os.path.exists(basedir):
		sys.exit('ERROR: no such basedir')
	
	for userID in userIDs:
		extractdir = basedir + userID + '/2-extracted/'	
		if not os.path.exists(extractdir):
			os.mkdir(extractdir)
	
	configs = []
	for device in devices:
		for mode in modes:
			for param in params:
				for is_restrictedsensors in is_restrictedsensorses:
					config = [basedir, userIDs, device, mode, param, True if 'TRUE' == is_restrictedsensors.upper() or '1' == is_restrictedsensors else False]
					configs.append(config)
	
	poolsize = min(maxpoolsize, poollimit, len(configs))
	print('Poolsize: ' + str(poolsize) + '  (Configs to run: ' + str(len(configs)) + ')')
	with Pool(poolsize) as p:
		p.map(process, configs)