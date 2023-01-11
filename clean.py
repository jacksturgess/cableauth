#python clean.py [basedir] [userID(s)]
# - basedir
# - userID(s) (may be a list)
#
#opens the file(s) in <basedir>/<userID>/0-raw/ and uses the <basedir>/event_timestamps.csv to extract the data for each gesture in a time window of up to <prewindowsizems> before the timestamp and <postwindowsizems> after
#outputs gestures-<device>.csv in <cleandir>/

import csv, math, os, re, shutil, sys
from _csv import Error as CSV_Error

basedir = sys.argv[1] + '/'
userIDs = re.split(',', (sys.argv[2]).lower())

#configs
sensors = ['Acc', 'Gyr', 'GRV', 'LAc']
minsamplespergesture = [200, 400] #watch, ring
prewindowsizems = 2000
postwindowsizems = 2000

#consts
w_s_index = 0 #sensor name
w_t_index = 1 #UNIX timestamp
w_x_index = 2
w_y_index = 3
w_z_index = 4
w_w_index = 5
r_t_index = 25 #UNIX timestamp
r_acc_x_index = 3
r_acc_y_index = 4
r_acc_z_index = 5
r_gyr_x_index = 0
r_gyr_y_index = 1
r_gyr_z_index = 2
r_lac_x_index = 20
r_lac_y_index = 21
r_lac_z_index = 22
r_grv_x_index = 14
r_grv_y_index = 15
r_grv_z_index = 16
r_grv_w_index = 13

def tidy_userIDs():
	global userIDs
	t_userIDs = []
	for userID in userIDs:	
		if 'user' in userID:
			t_userIDs.append('user' + f'{int(userID[4:]):03}')
		else:
			t_userIDs.append('user' + f'{int(userID):03}')
	t_userIDs.sort(reverse = False)
	userIDs = t_userIDs

if __name__ == '__main__':
	tidy_userIDs()
	
	if not os.path.exists(basedir):
		sys.exit('ERROR: no such basedir')
	
	eventsfile = basedir + 'event_timestamps.csv'
	if not os.path.exists(eventsfile):
		sys.exit('ERROR: no such eventsfile')
	
	#extract the event timestamps
	e_data = [] #container to hold all of the event timestamps
	with open(eventsfile, 'r') as f:
		data = list(csv.reader(f)) #returns a list of lists (each line is a list)
		data.pop(0) #removes the column headers
		counter = [0, 0, 0, 0]
		for datum in data:
			if len(datum) > 0: #disregards empty lines
				tag = datum[1]
				if 'Car unplugged' == tag:
					e_data.append([int(datum[0]), 'CAR_out-' + f'{counter[0]:04}'])
					counter[0] = counter[0] + 1
				elif 'Cable stowed at charger' == tag:
					e_data.append([int(datum[0]), 'CHARGER_in-' + f'{counter[1]:04}'])
					counter[1] = counter[1] + 1
				elif 'Cable unstowed from charger' == tag:
					e_data.append([int(datum[0]), 'CHARGER_out-' + f'{counter[2]:04}'])
					counter[2] = counter[2] + 1
				elif 'Car plugged in' == tag:
					e_data.append([int(datum[0]), 'CAR_in-' + f'{counter[3]:04}'])
					counter[3] = counter[3] + 1
	
	for userID in userIDs:
		sourcedir = basedir + userID + '/0-raw/'
		if not os.path.exists(sourcedir):
			sys.exit('ERROR: no such sourcedir for user: ' + userID)
		
		cleandir = basedir + userID + '/1-cleaned/'	
		if os.path.exists(cleandir):
			shutil.rmtree(cleandir)
		os.mkdir(cleandir)
		
		print("CLEANING DATA: " + userID)
		
		#open all watch/ring data files and extract gesture data by event timestamp
		for device in ['watch', 'ring']:
			w_data = [] #container to hold all of the watch/ring sensor data for this user
			
			(_, _, files) = next(os.walk(sourcedir))
			for file in files:
				filename = re.split('-', os.path.splitext(file)[0])
				if len(filename) > 3 and device == filename[2] and 'sensors' == filename[3] and '.csv' == os.path.splitext(file)[1]:
					with open(sourcedir + file, 'r') as f:
						d = list(csv.reader(f))
						if 'ring' == device:
							d.pop(0)
							
							#calculate and apply the time delta to convert local timestamps in the current ring sensor data file into UNIX time
							r_delta = 0
							r_deltafilename = sourcedir + filename[0] + '-' + filename[1] + '-ring-timesync.csv'
							if not os.path.exists(r_deltafilename):
								sys.exit('ERROR: no such timesync file: ' + r_deltafilename)
							with open(r_deltafilename, 'r') as f2:
								d2 = list(csv.reader(f2))
								r_delta = int((int(d2[0][2]) - int(d2[0][1])) / 1000)
							for datum in d:
								datum[r_t_index] = str(int(int(datum[r_t_index]) / 1000) + r_delta)
						w_data.append(d)
			
			w_outfilename = cleandir + 'gestures-' + device + '.csv'
			w_outfile = open(w_outfilename, 'w')
			w_outfile.write('GESTURE,SENSOR,ORIGINAL_TIMESTAMP,GESTURE_TIMESTAMP,X-VALUE,Y-VALUE,Z-VALUE,UNFILTERED_EUCLIDEAN_NORM')
			w_outfile.close()
			
			#for each timestamp: project a time window (backwards and forwards), grab the watch/ring sensor data inside that window, clean it, and write it
			for i in range(len(e_data)):
				triggertime = e_data[i][0]
				starttime = triggertime - prewindowsizems
				if i > 0 and triggertime > e_data[i - 1][0] and starttime < e_data[i - 1][0]:
					starttime = e_data[i - 1][0] #avoids window overlaps
				endtime = triggertime + postwindowsizems
				if i < len(e_data) - 1 and triggertime < e_data[i + 1][0] and endtime > e_data[i + 1][0]:
					endtime = e_data[i + 1][0] #avoids window overlaps
				
				#get gesture data
				g_data = [] #container to hold the gesture data
				g_is_found = False
				for data in w_data:
					if not g_is_found:
						index = r_t_index if 'ring' == device else w_t_index
						for datum in data:
							t = int(datum[index])
							if t >= starttime and t <= endtime:
								g_data.append(datum)
						if len(g_data) > 0:
							g_is_found = True
				
				#clean gesture data
				if g_is_found:
					g_output = []
					for datum in g_data:
						if 'watch' == device:
							s = datum[w_s_index]
							if s in sensors:
								t = datum[w_t_index]
								x = str('%.6f' % float(datum[w_x_index]))
								y = str('%.6f' % float(datum[w_y_index]))
								z = str('%.6f' % float(datum[w_z_index]))
								w = '0'
								norm = '0' #adds Euclidean norm of unfiltered values
								if 'GRV' == s:
									w = str('%.6f' % float(datum[w_w_index]))
								else:
									norm = str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z)))
								d = []
								d.append(s)
								d.append(t)
								d.append(str(float(int(t) - triggertime) / 1000)) #adds a normalised timestamp ending the gesture at 0 at the trigger point
								d.append(x)
								d.append(y)
								d.append(z)
								d.append(w)
								d.append(norm)
								g_output.append(d)
						elif 'ring' == device:
							t = datum[r_t_index]
							for s in sensors:
								if 'Acc' == s:
									x = str('%.6f' % float(datum[r_acc_x_index]))
									y = str('%.6f' % float(datum[r_acc_y_index]))
									z = str('%.6f' % float(datum[r_acc_z_index]))
									d = []
									d.append(s)
									d.append(t)
									d.append(str(float(int(t) - triggertime) / 1000))
									d.append(x)
									d.append(y)
									d.append(z)
									d.append('0')
									d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z)))) #adds Euclidean norm of unfiltered values
									g_output.append(d)
								elif 'Gyr' == s:
									x = str('%.6f' % float(datum[r_gyr_x_index]))
									y = str('%.6f' % float(datum[r_gyr_y_index]))
									z = str('%.6f' % float(datum[r_gyr_z_index]))
									d = []
									d.append(s)
									d.append(t)
									d.append(str(float(int(t) - triggertime) / 1000))
									d.append(x)
									d.append(y)
									d.append(z)
									d.append('0')
									d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
									g_output.append(d)
								elif 'GRV' == s:
									d = []
									d.append(s)
									d.append(t)
									d.append(str(float(int(t) - triggertime) / 1000))
									d.append(str('%.6f' % float(datum[r_grv_x_index])))
									d.append(str('%.6f' % float(datum[r_grv_y_index])))
									d.append(str('%.6f' % float(datum[r_grv_z_index])))
									d.append(str('%.6f' % float(datum[r_grv_w_index])))
									d.append('0')
									g_output.append(d)
								elif 'LAc' == s:
									x = str('%.6f' % float(datum[r_lac_x_index]))
									y = str('%.6f' % float(datum[r_lac_y_index]))
									z = str('%.6f' % float(datum[r_lac_z_index]))
									d = []
									d.append(s)
									d.append(t)
									d.append(str(float(int(t) - triggertime) / 1000))
									d.append(x)
									d.append(y)
									d.append(z)
									d.append('0')
									d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
									g_output.append(d)
					
					#output the gesture
					if len(g_output) >= minsamplespergesture[0 if 'watch' == device else 1]:
						w_output = []
						for datum in g_output:
							d = [e_data[i][1]]
							d.extend(datum)
							w_output.append(d)
						
						w_outfile = open(w_outfilename, 'a')
						w_outfile.write('\n' + '\n'.join([','.join(o) for o in w_output]))
						w_outfile.close()
					else:
						print(' Rejected: ' + e_data[i][1] + ' (' + str(len(g_output)) + ' samples)')
			print('OUTPUT: ' + w_outfilename)