def run_script(folder_name='dati_prova', subj_name='subj_01'):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 250)
    import datetime
    import os
    import math
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import welch, spectrogram
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from sklearn.preprocessing import MinMaxScaler
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    def conv_time(ns):
        date = datetime.datetime.fromtimestamp(ns*(10**-9))
        return date
    
    def convert_to_unix(timestamp_str):
        # Esempio di formato: '2024-01-26_09h47.19.968706'
        # Usiamo il metodo strptime per analizzare la stringa e ottenere un oggetto datetime
        dt = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %Hh%M.%S.%f')
    
        # Convertiamo l'oggetto datetime in un timestamp UNIX
        unix_timestamp_ns = int(dt.timestamp() * 1e9)  # Moltiplicato per 1 miliardo per convertire in nanosecondi
        return unix_timestamp_ns
    
    def conv_time(ms):
        ms = datetime.datetime.fromtimestamp(ms)
        return ms
    
    def convert_seconds_to_nanoseconds(seconds):
        nanoseconds = seconds * 1e9
        return nanoseconds
    
    def euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    events = pd.read_csv('./eyetracking_file/events.csv')
    events_list = np.arange(0,events.shape[0])
    
    for event in events_list:
        try:
        
            timestamp = events.at[event,'timestamp [ns]']
            rec_id = events.loc[event,'recording id']

            event = events.name[event]
        
            #subj_name = str('subj_03')
            gaze_mark = pd.read_csv('./eyetracking_file/gaze.csv')
            gaze_mark = gaze_mark.loc[(gaze_mark['timestamp [ns]'] > timestamp)]
            gaze_mark.reset_index(inplace=True)
            
            pupillometry_data  = pd.read_csv('./eyetracking_file/3d_eye_states.csv')
            pupillometry_data = pupillometry_data.loc[(pupillometry_data['timestamp [ns]'] > timestamp)]
            pupillometry_data.reset_index(inplace=True)
            fixations_mark = pd.read_csv('./eyetracking_file/fixations.csv')
            fixations_mark= fixations_mark.loc[(fixations_mark['start timestamp [ns]'] > timestamp)]
            fixations_mark.reset_index(inplace=True)
            blink  = pd.read_csv('./eyetracking_file/blinks.csv')
            blink = blink.loc[(blink['start timestamp [ns]'] > timestamp)]
            blink.reset_index(inplace=True)
            
            fixations_mark = fixations_mark.loc[(fixations_mark['recording id']==rec_id)]
            fixations_mark.reset_index(drop=True, inplace=True)
        
            gaze_mark = gaze_mark.loc[(gaze_mark['recording id']==rec_id)]
            gaze_mark.reset_index(drop=True,inplace=True)
            #gaze_mark.reset_index(drop=True,inplace=True)
            gaze_mark['fixation id'].fillna(-1,inplace=True)
            gaze_mark = gaze_mark.loc[(gaze_mark['gaze detected on surface'] == True)]
            gaze_mark.reset_index(drop=True,inplace=True)
            n_movement = 0
            for n,g in enumerate(range(gaze_mark.shape[0]-1)):
                if gaze_mark.at[n,'fixation id'] == -1:
                    n_gaze = n
                    while (n_gaze!=gaze_mark.shape[0]) and (gaze_mark.at[n_gaze,'fixation id']== -1):
                        gaze_mark.at[n_gaze,'fixation id'] = n+1+0.5
                        n_gaze+=1
            lista_movimenti = list()
            index = 0
            fixations_id_fatti = list()
            for n in range(gaze_mark.shape[0]-1):
                if (gaze_mark.at[n,'fixation id']%1 != 0) & (n==0) & (gaze_mark.at[n,'fixation id'] not in fixations_id_fatti):
                    fixations_id_fatti.append(gaze_mark.at[n,'fixation id'])
                    start = gaze_mark.at[n,'timestamp [ns]']
                    index = gaze_mark.at[n,'fixation id']
                    surface = gaze_mark.at[n,'gaze detected on surface']
                    n_gaze = n
                    first = True
                    lista_percorso = list()
                    while (gaze_mark.at[n_gaze,'fixation id']%1 != 0) or (n_gaze==gaze_mark.shape[0]):
                        end = gaze_mark.at[n_gaze,'timestamp [ns]']
                        if first == False:
                            x1 = gaze_mark.at[n_gaze-1,'gaze position on surface x [normalized]']
                            y1 = gaze_mark.at[n_gaze-1,'gaze position on surface y [normalized]']
                            x2 = gaze_mark.at[n_gaze,'gaze position on surface x [normalized]']
                            y2 = gaze_mark.at[n_gaze,'gaze position on surface y [normalized]']
                            spostamento = euclidean_distance(x1, y1, x2, y2)
                            lista_percorso.append(spostamento)
                        else:
                            start_x = gaze_mark.at[n_gaze,'gaze position on surface x [normalized]']
                            start_y = gaze_mark.at[n_gaze,'gaze position on surface y [normalized]']
                        first = False
                        n_gaze+=1
                    end_x = gaze_mark.at[n_gaze-1,'gaze position on surface x [normalized]']
                    end_y = gaze_mark.at[n_gaze-1,'gaze position on surface y [normalized]']
                    spostamento_tot = sum(lista_percorso)
                    spostamento_effettivo = euclidean_distance(start_x,start_y,end_x,end_y)
                    lista_movimenti.append([index,start,end,end-start,surface,spostamento_tot,(start_x,start_y),(end_x,end_y),spostamento_effettivo])
                elif (gaze_mark.at[n,'fixation id']%1 != 0) & (n!=0) & (gaze_mark.at[n,'fixation id'] not in fixations_id_fatti):
                    if index != gaze_mark.at[n-1,'fixation id']:
                        fixations_id_fatti.append(gaze_mark.at[n,'fixation id'])
                        start = gaze_mark.at[n,'timestamp [ns]']
                        index = gaze_mark.at[n,'fixation id']
                        surface = gaze_mark.at[n,'gaze detected on surface']
                        n_gaze = n
                        first = True
                        lista_percorso = list()
                        while (n_gaze!=gaze_mark.shape[0]) and (gaze_mark.at[n_gaze,'fixation id']%1 != 0):
                            end = gaze_mark.at[n_gaze,'timestamp [ns]']
                            if first == False:
                                x1 = gaze_mark.at[n_gaze-1,'gaze position on surface x [normalized]']
                                y1 = gaze_mark.at[n_gaze-1,'gaze position on surface y [normalized]']
                                x2 = gaze_mark.at[n_gaze,'gaze position on surface x [normalized]']
                                y2 = gaze_mark.at[n_gaze,'gaze position on surface y [normalized]']
                                spostamento = euclidean_distance(x1, y1, x2, y2)
                                lista_percorso.append(spostamento)
                            else:
                                start_x = gaze_mark.at[n_gaze,'gaze position on surface x [normalized]']
                                start_y = gaze_mark.at[n_gaze,'gaze position on surface y [normalized]']
                            first = False
                            n_gaze+=1
                        end_x = gaze_mark.at[n_gaze-1,'gaze position on surface x [normalized]']
                        end_y = gaze_mark.at[n_gaze-1,'gaze position on surface y [normalized]']
                        spostamento_tot = sum(lista_percorso)
                        spostamento_effettivo = euclidean_distance(start_x,start_y,end_x,end_y)
                        lista_movimenti.append([index,start,end,end-start,surface,spostamento_tot,(start_x,start_y),(end_x,end_y),spostamento_effettivo])
        
            lista_movimenti = pd.DataFrame(lista_movimenti,columns=['fixation id','start','end','duration','surface','spostamento_tot','spostamento_start','spostamento_end','spostamento_effettivo'])

            
            n_fixation = max(fixations_mark['fixation id']) 
            fixation_avg = fixations_mark['duration [ms]'].mean()
            fixation_std = fixations_mark['duration [ms]'].std()
            fixation_point_x = fixations_mark['fixation x [normalized]'].mean()
            fixation_point_x_std = fixations_mark['fixation x [normalized]'].std()
            fixation_point_y = fixations_mark['fixation y [normalized]'].mean()
            fixation_point_y_std = fixations_mark['fixation x [normalized]'].std()
        
            n_blink = blink.shape[0]
            blink_avg = blink['duration [ms]'].mean()
            blink_std = blink['duration [ms]'].std()
        
            pupillometry_start = pupillometry_data.reset_index().at[0,'pupil diameter left [mm]']
            pupillometry_end = pupillometry_data.at[pupillometry_data.shape[0]-1,'pupil diameter left [mm]']
            pupillometry_avg = pupillometry_data['pupil diameter left [mm]'].mean()
            pupillometry_std = pupillometry_data['pupil diameter left [mm]'].std()
        
            n_moviments = len(lista_movimenti)
            sum_time_movement = np.sum(lista_movimenti['duration'])/10**9
            avg_time_movement = np.mean(lista_movimenti['duration'])/10**9
            std_time_movement = np.std(lista_movimenti['duration'])/10**9
        
            gaze_fixation = list()
            for fixation in fixations_mark['fixation id'].loc[(fixations_mark['fixation id']%1==0)].to_numpy():
                gaze_fixation.append(gaze_mark.loc[(gaze_mark['fixation id']==fixation)].shape[0])
            n_gaze_fixation_avg = np.array(gaze_fixation).mean()
        
            gaze_movement = list()
            for fixation in lista_movimenti['fixation id'].to_numpy():
                gaze_movement.append(gaze_mark.loc[(gaze_mark['fixation id']==fixation)].shape[0])
            n_gaze_movement_avg = np.array(gaze_movement).mean()
        
            spostamento_totale_sum = (lista_movimenti['spostamento_tot']).sum()
            spostamento_totale_avg = (lista_movimenti['spostamento_tot']).mean()
            spostamento_totale_std = (lista_movimenti['spostamento_tot']).std()
        
            spostamento_effettivo_sum = (lista_movimenti['spostamento_effettivo']).sum()
            spostamento_effettivo_avg = (lista_movimenti['spostamento_effettivo']).mean()
            spostamento_effettivo_std = (lista_movimenti['spostamento_effettivo']).std()
        
            results = {
                'participant':subj_name,
                'n_fixation':n_fixation,
                'fixation_avg':fixation_avg,
                'fixation_std':fixation_std,
                'fixation_point_x':fixation_point_x,
                'fixation_point_x_std':fixation_point_x_std,
                'fixation_point_y':fixation_point_y,
                'fixation_point_y_std':fixation_point_y_std,
                'n_blink':n_blink,
                'blink_avg':blink_avg,
                'blink_std':blink_std,
                'pupillometry_start':pupillometry_start,
                'pupillometry_end':pupillometry_end,
                'pupillometry_avg':pupillometry_avg,
                'pupillometry_std':pupillometry_std,
                'n_moviments':n_moviments,
                'sum_time_movement':sum_time_movement,
                'avg_time_movement':avg_time_movement,
                'std_time_movement':std_time_movement,
                'n_gaze_fixation_avg':n_gaze_fixation_avg,
                'n_gaze_movement_avg':n_gaze_movement_avg,
                'spostamento_totale_sum':spostamento_totale_sum,
                'spostamento_totale_avg':spostamento_totale_avg,
                'spostamento_totale_std':spostamento_totale_std,
                'spostamento_effettivo_sum':spostamento_effettivo_sum,
                'spostamento_effettivo_avg':spostamento_effettivo_avg,
                'spostamento_effettivo_std':spostamento_effettivo_std
            }
        
            results = pd.DataFrame(results,index=[0])
            results.to_csv('./results_'+subj_name+'_'+str(event)+'.csv')
            ts = pupillometry_data['pupil diameter left [mm]'].to_numpy()
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.signal import welch, spectrogram
        
            # Genera dei dati di esempio
            fs = 200  # Frequenza di campionamento
            # Calcola il periodogramma utilizzando il metodo di Welch
            frequencies, periodogram = welch(ts, fs=fs, nperseg=100)
        
            # Stampa il periodogramma
            plt.figure(figsize=(10, 5))
            plt.semilogy(frequencies, periodogram)
            plt.title('Periodogram')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power spectral Density [V^2/Hz]')
            plt.grid(True)
            plt.savefig('./periodogram_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            # Calcola lo spettrogramma utilizzando il metodo di Welch
            f, t_spec, Sxx = spectrogram(ts, fs=fs, nperseg=256, noverlap=50)
        
            # Stampa lo spettrogramma
            plt.figure(figsize=(10, 5))
            plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.title('Spectrogram')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            plt.colorbar(label='Power [dB]')
            plt.savefig('./spectrogram_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            # pupillometry

            #%%
            fixations_copy = fixations_mark.copy()
            fixations_copy = fixations_copy.loc[(fixations_mark['fixation detected on surface'] == True)]
            fixations_copy.reset_index(drop=True,inplace=True)
            for row in range(pupillometry_data.shape[0]):
                for row2 in range(fixations_copy.shape[0]):
                    if ((pupillometry_data.at[row,'timestamp [ns]']>=fixations_copy.at[row2,'start timestamp [ns]']) and (pupillometry_data.at[row,'timestamp [ns]']<=fixations_copy.at[row2,'end timestamp [ns]'])):
                        pupillometry_data.at[row,'on surface'] = True
                        fixations_copy.drop(axis=0,labels=row2,inplace=True)
                        fixations_copy.reset_index(drop=True,inplace=True)
                        break
        
            print('fissazioni fatte')
        
            gaze_copy = gaze_mark.copy()
            gaze_copy = gaze_copy.loc[(gaze_copy['gaze detected on surface'] == True)]
            gaze_copy.reset_index(drop=True,inplace=True)
            for row in range(pupillometry_data.shape[0]):
                for row3 in range(gaze_copy.shape[0]):
                    if (pupillometry_data.at[row,'timestamp [ns]']==gaze_copy.at[row3,'timestamp [ns]']):
                        pupillometry_data.at[row,'on surface'] = True
                        gaze_copy.drop(axis=0,labels=row2,inplace=True)
                        gaze_copy.reset_index(drop=True,inplace=True)
                        break
        
            print('gaze fatto')
            plt.plot(pupillometry_data['pupil diameter left [mm]'])
            fig, ax = plt.subplots(figsize=(10,5))
        
            # Plot dei dati
            ax.plot(pupillometry_data['pupil diameter left [mm]'], marker='o', linestyle='-',markersize=1)
        
            # Colorazione dello sfondo in base alla colonna 'on surface'
            for idx, value in enumerate(pupillometry_data['on surface']):
                if value == 1:
                    ax.axvspan(idx - 0.5, idx + 0.5, facecolor='lightgreen', alpha=0.5)
                else:
                    ax.axvspan(idx - 0.5, idx + 0.5, facecolor='lightcoral', alpha=0.5)
        
            # Etichette e titolo
            ax.set_xlabel('Time')
            ax.set_ylabel('pupil diameter left [mm]')
            ax.set_title('Pupil Diameter with Background Indicator')
        
            plt.savefig('./pupil_surface_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            # HISTOGRAMS
        
            gaze = pd.read_csv('./eyetracking_file/gaze_not_enr.csv')
            gaze = gaze.loc[(gaze['timestamp [ns]'] > timestamp)]
            gaze.reset_index(inplace=True)
            
            fixations = pd.read_csv('./eyetracking_file/fixations.csv')
            fixations = fixations.loc[(fixations['start timestamp [ns]'] > timestamp)]
            fixations.reset_index(inplace=True)
            
            plt.hist(gaze['elevation [deg]'])
            plt.savefig('./hist_gaze_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            plt.hist(pupillometry_data['pupil diameter left [mm]'])
            plt.savefig('./hist_pupillometry_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            plt.hist(fixations['duration [ms]'])
            plt.savefig('./hist_fixations_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            plt.hist(blink['duration [ms]'])
            plt.savefig('./hist_blinks_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            # Saccades
        
            saccades  = pd.read_csv('./eyetracking_file/saccades.csv')
            saccades = saccades.loc[(saccades['start timestamp [ns]'] > timestamp)]
            saccades.reset_index(inplace=True)
            
            plt.hist(saccades['duration [ms]'])
            plt.savefig('./hist_saccades_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            # PATH GRAPH
        
            plt.plot(gaze['gaze x [px]'],gaze['gaze y [px]'], marker='o', linestyle='-', color='green')
            plt.savefig('./path_gaze_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            plt.plot(fixations['fixation x [normalized]'],fixations['fixation y [normalized]'], marker='o', linestyle='-', color='green')
            plt.savefig('./path_fixation_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
        
            for i in range(lista_movimenti.shape[0]):
                start_x = np.array(lista_movimenti.at[i, 'spostamento_start'][0])
                start_y = np.array(lista_movimenti.at[i, 'spostamento_start'][1])
                end_x = np.array(lista_movimenti.at[i, 'spostamento_end'][0])
                end_y = np.array(lista_movimenti.at[i, 'spostamento_end'][1])
        
                plt.scatter(start_x, start_y, marker='o', c='b')
                plt.scatter(end_x, end_y, marker='o', c='b')
        
                plt.plot([start_x, end_x], [start_y, end_y], linestyle='-', c='b')
        
            plt.savefig('./total_mov_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            shape = lista_movimenti.shape[0]-1
            scatter_list_x = [np.array(lista_movimenti.at[0,'spostamento_start'][0]), np.array(lista_movimenti.at[shape,'spostamento_end'][0])]
            scatter_list_y = [np.array(lista_movimenti.at[0,'spostamento_start'][1]), np.array(lista_movimenti.at[shape,'spostamento_end'][1])]
            
            # SCATTER PLOT
            plt.scatter(scatter_list_x, scatter_list_y, marker='o')
            
            # LINE
            plt.plot(scatter_list_x, scatter_list_y, linestyle='-', color='blue')
            plt.savefig('./effective_mov_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
        
            # CLOUD POINT
        
            scale = 1000
            fix_x = fixations['fixation x [normalized]']*scale
            fix_y = fixations['fixation y [normalized]']*scale
        
            kde = gaussian_kde([fix_x,fix_y])
            x_grid, y_grid = np.mgrid[0:scale:scale*1j, 0:scale:scale*1j]
            z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()]))
            plt.figure(figsize=(8, 4))
            plt.contourf(x_grid, y_grid, z.reshape(x_grid.shape), cmap='Reds', alpha=0.7)
            plt.colorbar()
            plt.scatter(fix_x,fix_y, alpha=0.4)
            #plt.imshow(np.rot90(background,k=2))
            plt.xlim(0,scale)
            plt.ylim(0,scale)
            plt.savefig('./cloud_fix_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            scale = 1000
            gaze_x = gaze_mark['gaze position on surface x [normalized]']*scale
            gaze_y = gaze_mark['gaze position on surface y [normalized]']*scale
        
            kde = gaussian_kde([gaze_x,gaze_y])
            x_grid, y_grid = np.mgrid[0:scale:scale*1j, 0:scale:scale*1j]
            z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()]))
            plt.figure(figsize=(8, 4))
            plt.contourf(x_grid, y_grid, z.reshape(x_grid.shape), cmap='Reds', alpha=0.7)
            plt.colorbar()
            plt.scatter(gaze_x,gaze_y, alpha=0.4)
            #plt.imshow(np.rot90(background,k=2))
            plt.xlim(0,scale)
            plt.ylim(0,scale)
            plt.savefig('./cloud_gaze_'+subj_name+'_'+str(event)+'.pdf')
            plt.close()
        
            # Verifica che saccades non sia vuoto
            if saccades.shape[0] != 0:
                
                # Conversione dei timestamp da nanosecondi a microsecondi
                saccades['start timestamp [ns]'] /= 1000
                saccades['end timestamp [ns]'] /= 1000
                saccades['start timestamp [ns]'] = saccades['start timestamp [ns]'].astype('int')
                saccades['end timestamp [ns]'] = saccades['end timestamp [ns]'].astype('int')
                
                # Calcola il tempo minimo e massimo
                min_time = saccades['start timestamp [ns]'].min()
                max_time = saccades['end timestamp [ns]'].max()
                
                # Crea la serie temporale dei saccades
                time_series = np.zeros(max_time + 1 - min_time)
                for _, row in saccades.iterrows():
                    time_series[row['start timestamp [ns]'] - min_time : row['end timestamp [ns]'] - min_time + 1] = 1
                
                # Scala la serie temporale a un massimo di 1000 punti
                max_points = 1000
                if len(time_series) > max_points:
                    factor = len(time_series) // max_points
                    time_series = time_series[::factor]
                    time = np.arange(len(time_series)) * factor
                else:
                    time = np.arange(len(time_series))
                
                # Plot della serie temporale ridotta
                fig, ax = plt.subplots(figsize=(20, 5))
                ax.plot(time, time_series, drawstyle='steps-post', marker='o', markersize=2)
                ax.set_xlabel('Time')
                ax.set_ylabel('Blink (0 = No, 1 = Yes)')
                ax.set_title('Time Series of Saccades')
                plt.savefig('./blink_'+subj_name+'_'+str(event)+'.pdf', dpi=72)
                plt.close()
                
                # Plot di amplitude e velocity
                plt.plot(saccades['amplitude [px]'])
                plt.savefig('./amplitude_saccades_'+subj_name+'_'+str(event)+'.pdf', dpi=72)
                plt.close()
                
                plt.plot(saccades['mean velocity [px/s]'])
                plt.plot(saccades['peak velocity [px/s]'])
                plt.savefig('./velocity_saccades_'+subj_name+'_'+str(event)+'.pdf', dpi=72)
                plt.close()
        except:
            print('error in ',event)

    def downsample_video(input_file, output_file, input_fps, output_fps):
        cap = cv2.VideoCapture(input_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, output_fps, (width, height))

        frame_interval = int(input_fps / output_fps)

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_interval == 0:
                out.write(frame)

        cap.release()
        out.release()

    downsampled_video_file = './eyetracking_file/downsampled_video2.mp4'
    downsample_video('./eyetracking_file/internal.mp4', downsampled_video_file, 200, 40)
    pupillometry_data  = pd.read_csv('./eyetracking_file/3d_eye_states.csv')
    time_series = pupillometry_data['pupil diameter left [mm]'].values.flatten()

    video_file1 = downsampled_video_file
    cap1 = cv2.VideoCapture(video_file1)

    video_file2 = './eyetracking_file/external.mp4'
    cap2 = cv2.VideoCapture(video_file2)

    fig, (video_axes1, video_axes2, time_series_axes) = plt.subplots(3, 1, figsize=(10, 8))

    ret1, frame1 = cap1.read()
    frame_height1, frame_width1, _ = frame1.shape

    ret2, frame2 = cap2.read()
    frame_height2, frame_width2, _ = frame2.shape

    video_axes1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    video_axes1.axis('off')
    video_axes2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    video_axes2.axis('off')

    window_size = 1000
    initial_idx = np.arange(window_size)
    time_series_plot, = time_series_axes.plot(initial_idx, time_series[initial_idx], 'b')
    time_series_axes.set_xlabel('Frames (n)')
    time_series_axes.set_ylabel('Diameter (mm)')
    ball, = time_series_axes.plot([1], [time_series[0]], 'ro', markersize=10)
    window_line, = time_series_axes.plot([0, window_size], [time_series[window_size // 2]] * 2, 'b--')

    output_file = 'output_video.mp4'
    fps = cap1.get(cv2.CAP_PROP_FPS)
    output_size = (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, output_size)

    def update(frame):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            return

        video_axes1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        video_axes2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        i = frame
        ball.set_data([i], [time_series[i]])

        idx_start = max(0, i - window_size // 2)
        idx_end = min(len(time_series), i + window_size // 2)
        idx = np.arange(idx_start, idx_end)

        time_series_plot.set_data(idx, time_series[idx])
        window_line.set_data([idx_start, idx_end], [time_series[i]] * 2)

        fig.canvas.draw()

        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out.write(img)

        return video_axes1, video_axes2, time_series_plot, ball, window_line

    num_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    ani = FuncAnimation(fig, update, frames=range(num_frames), interval=1000 / fps)

    for frame in range(0,num_frames):
        update(frame)

    cap1.release()
    cap2.release()
    out.release()