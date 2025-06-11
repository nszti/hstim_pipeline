import math
import serial
import logging
import subprocess
import time


class TransmitTrial:
    def __init__(self, serialPortName):
        '''
        Class to transmit a trial to the Arduino (and receive information back)

        This class consists mainly out of technical functions to communicate with the Arduino and transmit all the
        correct information. No changes needed (changing it may lead to the Arduino not responding in the correct way
        and therefore not starting any trial).

        Timeout is initially set to 5 seconds.

        '''

        baudrate = 115200
        self.nb_levels = 1
        try:
            self.arduino = serial.Serial(port=serialPortName,baudrate=baudrate,rtscts=True,timeout=30, write_timeout=5) # timeout set to 30 seconds, which ensures enough time for the ttl-pulse
        except serial.SerialException as e:
            raise ConnectionError('Arduino is not connected to the right port. Chosen output port is ' + str(serialPortName), e)

        # Wait until Arduino is configured and serial link is initialized (and check if baudrate is the same)
        logging.debug('Baudrate=' + str(baudrate) + '. Nb_levels =' + str(self.nb_levels))
        logging.debug('Waiting for Arduino to reset')
        msg = self.recvStringFromArduino()
        if msg == 'Arduino is ready. Baudrate=' + str(baudrate) + '. Nb_levels=' + str(self.nb_levels):
            logging.debug('Successful serial connection with Arduino.')
            return
        raise RuntimeError('Incorrect reply from Arduino')


    def transmit_byte_array(self, chip, arr):
        '''
        Transmit an array of bytes to the arduino and check if it has arrived
        '''

        if not isinstance(arr, bytearray):
            raise Exception('Error: invalid type to be transmitted (bytearray needed).')
        self.arduino.write(arr)

        rec_first_byte = int(self.readLineFromArduino())
        logging.debug('First byte for ' + chip + ': ' + str(rec_first_byte))
        rec_last_byte = int(self.readLineFromArduino())
        logging.debug('Last byte for ' + chip + ': ' + str(rec_last_byte))

        if rec_first_byte != arr[0] or rec_last_byte != arr[-1]:
            raise Exception('Error: bytes from ' + str(chip) + ' are inccorrectly transmitted.')

    def transmit_integer_array(self,integer_array):
        '''
        Transmit an array of integers to the Arduino and check if it has arrived
        '''

        if not isinstance(integer_array, list):
            raise Exception('Error: invalid type given to be transmitted.')
        if not isinstance(integer_array[0], int):
            raise Exception('Error: invalid type to be transmitted (integers needed).')

        transmit_timing = ';'.join(str(elem) for elem in integer_array)
        self.arduino.write(transmit_timing.encode())

        rec_first_value = int(self.readLineFromArduino())
        logging.debug('First integer value: ' + str(rec_first_value))
        rec_last_value = int(self.readLineFromArduino())
        logging.debug('Last integer value: ' + str(rec_last_value))

        if rec_first_value != integer_array[0] or rec_last_value != integer_array[-1]:
            raise Exception('Error: values are inccorrectly transmitted.')


    def transmit_nb(self,nb):

        self.sendStringToArduino(str(nb))
        arduinoReply = self.recvStringFromArduino()
        if not int(arduinoReply) == nb:
            raise Exception('Error: string transmission. Sent: ' + str(nb) + '. Received: ' + arduinoReply)


    def transmit_string(self,string):
        '''
        Sends a string to the Arduino and receives it back as confirmation
        '''

        if len(string) != 11:
            raise Exception('Error: transmitted string should contain 11 characters.')

        self.sendStringToArduino(string)
        arduinoReply = self.recvStringFromArduino()
        if not arduinoReply == string:
            raise Exception('Error: string transmission. Sent: ' + str(string) + '. Received: ' + str(arduinoReply))



    def receive_values(self):
        '''
        Receives information from the Arduino
        Returns a list with the received values (float)
        '''
        received_values = []
        logging.debug('Start receiving values from Arduino')
        while(1):
            ans = self.readLineFromArduino()
            if 'value_transmission_ended' in ans:
                break
            received_values = received_values + [float(ans)]
        logging.debug('Finished receiving values from Arduino')
        return received_values


    def sendStringToArduino(self, stringToSend):
        '''
        Sends a string to the Arduino
        '''
        startMarker = '<'
        endMarker = '>'

        stringWithMarkers = (startMarker)
        stringWithMarkers += stringToSend
        stringWithMarkers += (endMarker)

        logging.debug('String sent: ' + stringWithMarkers)
        self.arduino.write(stringWithMarkers.encode('utf-8'))  # encode needed for Python3


    def recvStringFromArduino(self):
        '''
        Receives a string from the Arduino or raises TimeoutError.
        '''
        startMarker = '<'
        endMarker = '>'
        dataBuf = None

        while True:
            x = self.arduino.read().decode("utf-8")  # decode needed for Python3
            if not x:
                logging.debug('Partial string received: ' + dataBuf)
                raise TimeoutError('Received no string from Arduino')
            if dataBuf is not None:
                if x != endMarker:
                    dataBuf = dataBuf + x
                else:
                    logging.debug('String received: ' + dataBuf)
                    return dataBuf
            elif x == startMarker:
                dataBuf = ''


    def readLineFromArduino(self):
        '''
        Receives a newline-terminated string from the Arduino or raises TimeoutError.
        '''
        line = self.arduino.readline().decode()
        if not line:
            raise TimeoutError('Received nothing from Arduino')
        return line.rstrip()


    def readLinesFromArduino(self, timeout):
        '''
        Receives newline-terminated strings from the Arduino or raises TimeoutError.
        Note the timeout is only an approximation.
        '''
        maxAttempts = int(math.ceil(timeout / self.arduino.timeout)) or 1
        attempts = 0
        while attempts < maxAttempts:
            line = self.arduino.readline().decode()
            if not line:
                attempts += 1
                continue
            yield line.rstrip()
        raise TimeoutError('Received no line from Arduino')


    def stimulation(self, trial, wait_until_done=True):
        '''
        Send all necessary information about a stimulation trial to the Arduino, that will then be executed.
        wait_until_done determines whether wait_until_stimulation_done() gets called automatically or not.
        '''
        if trial.nb_levels != self.nb_levels:
            raise Exception('Error: Arduino expects ' + str(self.nb_levels) + ' levels, but sequences were made for ' + str(trial.nb_levels) + ' levels.')

        self.transmit_string('stimulation')
        self.transmit_nb(trial.nb_commands)
        self.transmit_nb(trial.nb_toggles)
        self.transmit_nb(trial.nb_gnds)
        self.transmit_byte_array('switch',trial.switch_array)
        if trial.stimulator_timing != sorted(trial.stimulator_timing):
            raise Exception('Stimulator timing is not sorted.')
        self.transmit_integer_array(trial.stimulator_timing)
        self.transmit_integer_array(trial.toggle_array)
        self.transmit_integer_array(trial.gnd_array)
        self.transmit_byte_array('stimulator',trial.stimulator_array)
        self.transmit_nb(trial.nb_repeats)            # specifically for calcium imaging
        self.transmit_nb(trial.trial_delay)           # specifically for calcium imaging
        self.transmit_nb(int(trial.pulse_shorting_delay))  # specifically for calcium imaging
        self.transmit_string('*confirmed*')
        logging.info('Stimulation trial started.')
        if wait_until_done:
            self.wait_until_stimulation_done(trial.duration() / 1000000)


    def wait_until_stimulation_done(self, timeout):
        '''
        For use after (or in) stimulation(): receives whatever the Arduino sends back until
        'stimulation_ended' is seen which indicates stimulation finished.
        Raises TimeoutError if this does not happen within the given timeout.
        '''
        for ans in self.readLinesFromArduino(timeout):
            if 'stimulation_ended' in ans:
                break

            if 'tdcs_ended' in ans:
                break
            logging.info(ans)

        logging.info('Stimulation trial finished.')

    def impedance(self):
        '''
        Let Arduino know it must execute an impedance measurement, and receive the values.
        The exact commands for this are coded on the Arduino, because this doesn't require any user input.

        The stimulator loops over all electrodes and connects them 1 by 1 to channel 1A on that level.
        Stimulate 10 biphasic pulses with 1.4 uA, measure peak-to-peak-voltage and take average over the 10 pulses.

        Values received are in kOhm.
        '''
        self.transmit_string('impedance**')
        logging.info('Impedance testing started.')
        while(1):
            ans = self.readLineFromArduino()
            if 'start_sending' in ans:
                break
            logging.debug(ans)
        logging.info('Impedance measurement finished.')
        return self.receive_values()

    def calibration(self):
        self.transmit_string('calibration')
        logging.info('Calibration started.')
        while(1):
            ans = self.readLineFromArduino()
            if 'calibration_ended' in ans:
                break
            logging.debug(ans)
        logging.info('Calibration finished')

    def waveform(self):
        '''
        Let Arduino know it must execute a waveform measurement, and receive the values.
        The exact commands for this are coded on the Arduino
        This function is not needed for the normal functioning of trials
        '''
        self.transmit_string('waveform***')
        logging.info('Waveform measurement started.')
        while(1):
            ans = self.readLineFromArduino()
            if 'start_sending' in ans:
                break
            logging.debug(ans)
        logging.info('Waveform measurement finished.')
        return self.receive_values()

    def tdcs(self, amplitude=1.1, duration=60, rest=1, wait_until_done=True):
        '''
        Send all necessary information about a tDCS trial to the Arduino, that will then be executed.
        wait_until_done determines whether wait_until_stimulation_done() gets called automatically or not.
        '''

        self.tdcs_amplitude = int(amplitude*1000) # For arduino, should be in µA
        self.tdcs_duration = int(duration*1000) # For arduino, should be in ms
        self.tdcs_rest = int(rest*1000) # For arduino, should be in ms


        self.transmit_string('tdcs*******')
        self.transmit_integer_array([self.tdcs_amplitude, self.tdcs_duration, self.tdcs_rest])
        self.transmit_string('*confirmed*')
        logging.info('Stimulation trial started.')
        if wait_until_done:
            self.wait_until_stimulation_done(1 + (self.tdcs_duration + self.tdcs_rest) / 1000)





class ConfigureTrial:
    def __init__(self, trial_array=None, gnd_electrodes=[], nb_repeats=1, trial_delay=8000, interleaved=False, pulse_shorting_delay=3470, min_frequency=50):
        '''
        Class to create the correct stimulator and switch configurations based on high-level information about the pulses

        All time parameters are in microseconds, all amplitude parameters are in microamperes

        Feel free to create extra methods to create trial_arrays, as long as they have the following form:
        trial_array = [electrode, amplitude_cath, start_t_cath, duration_cath, amplitude_an, start_t_an, duration_an]
        '''
        self.interleaved = interleaved

        for pulse in range(0, len(trial_array)):
            trial_array[pulse][0] = self.remap([trial_array[pulse][0]])[0]

        self.trial_array = trial_array
        self.nb_repeats = nb_repeats
        if min_frequency < 5:
            raise Exception('Minimal chosen frequency is lower than 5Hz, please choose a larger frequency.')
        self.min_frequency = min_frequency # in Hz
        if not trial_delay >= 500:
            raise Exception('Trial delay has to be 500ms or larger for Arduino code to function.')
        self.trial_delay = trial_delay
        self.pulse_shorting_delay = pulse_shorting_delay # 3470 is default case for symmetric biphasic pulse of 200µs and 200Hz
        # If shorting delay between pulses is longer, the desired frequency of the pulses wont be reached
        # If shorting delay between pulses is shorter than allowed, there might be some residual DC

        if self.interleaved:
            for _ in gnd_electrodes:
                _.append(1)
            self.gnd_electrodes = self.remap_list(gnd_electrodes) ## remap added on 7 oct 22 for 2p --> still check if necessary and if mapping is now ok
        else:
            gnd_electrodes.append(1)
            self.gnd_electrodes = self.remap(gnd_electrodes)
        self.rec_electrodes = None
        # self.rec_electrodes = self.remap([17]) # manually choose a second electrode to connect with the external oscilloscope
        # Using the record-channel leads to DC-issues on certain PCBs. If use of it is necessary, make sure to ground everything right before and after (see execute_impedance in Arduino).
        self.rec_electrodes = self.remap([1]) # Connect recording-channel to a non-existing electrode to ground it.

        self.nb_levels = 1

        self.assign_channels()
        self.configure_switches()
        self.configure_stimulators()

        self.nb_commands = len(self.stimulator_timing)

        logging.info('Trial configuration finished.')



    @classmethod
    def from_pulses(cls, trial_array, gnd_electrodes=[]):
        '''
        Create a trial object based on individual pulses
        Important: electrodes need to start counting from 0 (necessary for later calculations)
        '''
        return cls(trial_array, gnd_electrodes)

    @classmethod
    def from_bipolar_train(cls, dict, gnd_electrodes=[], nb_repeats=1, trial_delay=8000, interleaved=False):
        '''
        Create a trial object based on higher-level information about the needed pulse trains
        Important: electrodes need to start counting from 0 (necessary for later calculations)
        '''
        pulse_shorting_delay = 100000
        min_frequency = 800 # Take something high so you always get the actual minimal frequency instead of the default value

        trial_array = []
        for train in range(0,len(dict)):
            min_frequency = min(min_frequency, dict[train]['frequency'])
            for pulse in range(0,dict[train]['nb_pulses']):
                start = int(dict[train]['start'] + 2800 + pulse/dict[train]['frequency']*1000000) # The 2800 is needed to allow some time for configuring the DACs before a stimulation pulse. Previously, this 2800 was set at the bottom of the code.
                trial_array.append([dict[train]['electrode'],dict[train]['amplitude'],start, dict[train]['duration'], -dict[train]['amplitude']/dict[train]['multiplier'],start+dict[train]['duration']+dict[train]['interpulse interval'],dict[train]['duration']*dict[train]['multiplier']])
                # trial_array = [electrode, amplitude_cath, start_t_cath, duration_cath, amplitude_an, start_t_an, duration_an]
                pulse_shorting_delay = min(pulse_shorting_delay, -2130 + 1/dict[train]['frequency']*1000000 - dict[train]['duration'] - dict[train]['interpulse interval'] - dict[train]['duration']*dict[train]['multiplier'])

        return cls(trial_array, gnd_electrodes, nb_repeats, trial_delay, interleaved, pulse_shorting_delay, min_frequency)

    def remap(self, elec):
        '''
        Remap the electrodes. In the python code, electrodes start counting from 0.
        Also, the connectors are upside down, so 32 is 31, and 31 is 32.
        '''
        if elec is not None:
            for _ in range(0,len(elec)):
                elec[_] -= 1

        return elec

    def remap_list(self, lst):
        '''
        Remap a list of lists.
        '''
        for elem in lst:
            elem = self.remap(elem)

        return lst


    def duration(self):
        '''
        Duration of complete trial, as offset + duration of last pulse.
        + allow some time for the last commands to be processed (like zero-volt period)
        '''
        return max(t[-2] + t[-1] for t in self.trial_array) + 10000


    def configure_switches(self):
        '''
        Create the command sequence for the analog switch arrays
        '''

        self.nb_switches = self.nb_levels*6
        self.nb_switch_bytes = int(32*self.nb_switches)          # 256/8
        self.switch_array = bytearray(self.nb_switch_bytes)         # all connections open (0) by default
        self.toggle_array = []

        connections = [[], []]

        # Connect stimulator channels to electrodes

        for pulse in range(0,len(self.trial_array)):

            # Calculate all necessary parameters
            elec = self.trial_array[pulse][0]
            switch = elec // 16
            lvl = switch // 6
            ch = self.trial_array[pulse][1] - lvl * 16

            if ch < 12:
                y = ch
            elif ch < 14 and switch % 6 < 4:
                y = ch
            elif ch >= 13 and switch % 6 >= 4:
                y = ch - 2
            else:
                raise Exception('Combination of electrode ' + str(elec) + ' with stimulator channel ' + str(
                    ch + lvl * 16) + ' is not possible.')

            x = elec % 16
            connection_byte = (self.nb_levels * 6 - 1 - switch) * 32 + 2 * (15 - y) + (
                        15 - x) // 8  # this last part has changed
            # connection_byte = (switch) * 32 + 2 * (15 - y) + x // 8
            connection = x % 8

            if not self.trial_array[pulse][0] in connections[0]:
                self.switch_array[connection_byte] += 2**connection

                connections[0].append(elec)
                connections[1].append(ch+lvl*16)
                # Add the connection bytes and bits to the toggle_array
                connection_byte = (self.nb_levels * 6 - 1 - switch) * 32 + (15 - elec % 16) // 8 # connect this electrode to ground
                self.toggle_array.extend([connection_byte])
                self.toggle_array.extend([connection])

            # If electrode already in definitions but from previous pulse train: add byte and bit
            elif self.trial_array[pulse][3] > [l[6]+l[7]+ 20000 for l in self.trial_array[0:pulse] if l[1] == self.trial_array[pulse][1]][-1]: # same definition as first_pulse
                # Add the connection bytes and bits to the toggle_array
                connection_byte = (self.nb_levels * 6 - 1 - switch) * 32 + (15 - elec % 16) // 8 # connect this electrode to ground
                self.toggle_array.extend([connection_byte])
                self.toggle_array.extend([connection])


        logging.debug("Connections between electrodes (first list) and stim channels (second list): " + str(connections))


        # Connect GND (Y15) to electrodes
        if self.interleaved == False: # In the non-interleaved case, there is just a list of gnd_electrodes.
            for gnd_elec in self.gnd_electrodes:
                if gnd_elec in connections[0]:
                    raise Exception('Electrodes connected for stimulation cannot be ground: ' + str(gnd_elec))
                switch = gnd_elec // 16
                connection_byte = (self.nb_levels*6-1-switch)*32 + (15 - gnd_elec%16)//8
                #connection_byte = (switch) * 32 + gnd_elec // 8
                connection = gnd_elec%8
                self.switch_array[connection_byte] += 2**connection

        # Connect RECORD (Y14) to electrodes
        if self.rec_electrodes is not None:
            connected_rec_levels = []
            for rec_elec in self.rec_electrodes:
                switch = rec_elec // 16
                lvl = switch // 6
                if lvl in connected_rec_levels:
                    raise Exception('Only 1 rec_elec per level is possible: ' + str(self.rec_electrodes))
                else:
                    connected_rec_levels.append(lvl)
                connection_byte = (self.nb_levels*6-1-switch)*32 + 2 + (15 - rec_elec%16)// 8
                #connection_byte = (switch) * 32 + 2 + rec_elec // 8
                connection = rec_elec % 8
                self.switch_array[connection_byte] += 2 ** connection


    def configure_stimulators(self): # a lot changed in this function
        '''
        Create the command sequence for the stimulator chips, including the order and timing in which each command has to be executed
        Note: the cathodic (first phase) and anodic (first phase) have been changed into first and second, to allow both cathodic-first and anodic-first stimulation.
        '''

        self.nb_stimulator_bytes = 3*4*self.nb_levels
        self.stimulator_array = bytearray()
        self.stimulator_array_list = []
        self.stimulator_timing = []
        self.gnd_array = []



        zero_volt_period = 500000  # in microseconds
        charge_calibration = [1]*self.nb_levels*4*4
        # Manual charge calibration --> TO DO: test all channels
        # Attention: start counting from stim 1 on lvl 1 (ch A to D), then stim 2 on lvl 2 (ch A to D) = reversed order than on Arduino
        #charge_calibration[0] = 1.5 # output ch 1-A on lvl 0 (lowest level)
        #charge_calibration[16] = 0.75 # output ch 1-A on lvl 1

        inter_pulse_period = int(1/self.min_frequency*10**6) + 1000 # in us

        train_counter = 0
        for pulse in range(0,len(self.trial_array)):
            ch = self.trial_array[pulse][1]

            first_pulse = False
            last_pulse = False

            if pulse == 0:
                first_pulse = True
            elif ch not in [l[1] for l in self.trial_array[0:pulse]]:
                first_pulse = True
            # If end of pulse is more than inter_pulse_period away from next pulse, consider it a next train
            # Note: if zero_volt_period applies, depends on the "last_pulse" rather than on the "first_pulse"
            elif self.trial_array[pulse][3] > [l[6]+l[7] + max(inter_pulse_period,zero_volt_period) for l in self.trial_array[0:pulse] if l[1] == ch][-1]:
                first_pulse = True

            if pulse == len(self.trial_array)-1:
                last_pulse = True
            elif ch not in [l[1] for l in self.trial_array[pulse+1:]]:
                last_pulse = True
            # If end of pulse is more than inter_pulse_period away from next pulse, consider it a next train
            elif self.trial_array[pulse][6] + self.trial_array[pulse][7] + max(inter_pulse_period,zero_volt_period) < [l[3] for l in self.trial_array[pulse+1:] if l[1] == ch][0]:
                last_pulse = True


            amp_first = self.trial_array[pulse][2]
            start_first = self.trial_array[pulse][3]
            duration_first = self.trial_array[pulse][4]

            amp_second = self.trial_array[pulse][5]
            start_second = self.trial_array[pulse][6]
            duration_second = self.trial_array[pulse][7]

            HiZ_current = 0x8000 # calibration on the Arduino ensures this is indeed 0uA

            # Do same basic checks on the pulse
            if not -24000 < amp_first < 24000:
                raise Exception('Choose first amplitude between -24000 and 24000 microamps, for pulse ' + str(pulse))
            if not -24000 < amp_second < 24000:
                raise Exception('Choose second amplitude between -24000 and 24000 microamps, for pulse ' + str(pulse))
            if not round(-amp_first*duration_first) == round(amp_second*duration_second):
                raise Exception('No charge balancing for pulse ' + str(pulse))
            if not start_first + duration_first <= start_second:
                raise Exception('Overlapping phases for pulse ' + str(pulse))
            if not start_second + duration_second < 2**32-5000000: # max value for unsigned long on Arduino minus 5 seconds
                raise Exception('Trial should be less than 70min')


            # Charge balancing --> make 2nd phase a little bit longer
            duration_second = int(duration_second*charge_calibration[ch])

            counter_artefact = True
            amp_counter_artefact = 20 # in uA

            amp_first = int((amp_first+24000)*65536/48000)
            amp_second = int((amp_second+24000)*65536/48000)
            amp_counter_artefact = int((amp_counter_artefact+24000)*65536/48000)

            stim = ch//4
            ch = ch%4
            ch_byte = 1 << (5 + ch)

            if first_pulse:
                before_period = 2800

                self.write_to_register(stim, 0x03, ch_byte, start_first - before_period)

                # Command: select range (previously this also included code for slew rate but this worked countereffective)
                self.write_to_register(stim, 0x04, 0x0007, start_first - before_period)

                # Command: set amplitude
                self.write_to_register(stim, 0x05, HiZ_current, start_first - before_period)

                self.write_to_register(stim, 0x04, 0x1007, start_first - before_period)

                self.write_to_register(stim, 0x05, HiZ_current, start_first - before_period)

                self.write_to_register(stim, 0x05, HiZ_current, start_first - before_period + 1) # To make sure that the end of the "shorting step" happens after this

                if counter_artefact:
                    # Short positive pulse to counter-act the artefact
                    counter_pulse_period = 150

                    #self.write_to_register(stim, 0x05, amp_counter_artefact, start_cath - before_period)

                    #self.write_to_register(stim, 0x03, ch_byte, start_cath - before_period + counter_pulse_period)

                    self.write_to_register(stim, 0x05, HiZ_current, start_first - before_period + counter_pulse_period)

                self.toggle_array.insert(train_counter*4,start_first - before_period)
                self.toggle_array.insert(train_counter*4+1,start_first - before_period + 1) # change this time and the value of 800: time how long setting of the switches takes



            # Command: set amplitude
            self.write_to_register(stim, 0x03, ch_byte, start_first)

            self.write_to_register(stim, 0x05, amp_first, start_first)

            if start_first+duration_first < start_second:

                # Command: select channel
                self.write_to_register(stim, 0x03, ch_byte, start_first + duration_first)

                # Command: disable output (HiZ)
                self.write_to_register(stim, 0x05, HiZ_current, start_first + duration_first)

            # Command: select channel
            self.write_to_register(stim, 0x03, ch_byte, start_second)

            # Command: set amplitude
            self.write_to_register(stim, 0x05, amp_second, start_second)

            if not last_pulse:

                # Command: select channel
                self.write_to_register(stim, 0x03, ch_byte, start_second + duration_second)

                # Command: set HiZ
                self.write_to_register(stim, 0x05, HiZ_current, start_second + duration_second)

                self.gnd_array.append([start_second + duration_second, self.trial_array[pulse][0]]) # [time, electrode]

                if self.interleaved == True: # If self.interleaved == False, all of the ground electrodes are already constantly grounded anyway
                    for gnd_electrode in self.gnd_electrodes[train_counter]:
                        self.gnd_array.append([start_second + duration_second, gnd_electrode]) # [time, electrode]

            # Command: set amplitude
            #self.write_to_register(stim, 0x05, 0x8000, start_an + duration_an + zero_volt_period)

            #self.write_to_register(stim, 0x04, 0x1007, start_an + duration_an + zero_volt_period)

            elif last_pulse:
                self.write_to_register(stim, 0x03, ch_byte, start_second + duration_second)

                # Command: set at 0V
                self.write_to_register(stim, 0x04, 0x1000, start_second + duration_second)

                self.write_to_register(stim, 0x03, ch_byte, start_second + duration_second + zero_volt_period)

                # Command: disable output (HiZ)
                self.write_to_register(stim, 0x04, 0x0000, start_second + duration_second + zero_volt_period)

                train_counter = train_counter + 1




        # First concatenate elements from
        # Sort list of bytearray, based on the timing array
        # MicroPython's builtin sort is not stable, so instead use a simple bubble sort.
        def bubble_sort(ar, key):
            n = len(ar)
            for i in range(n - 1):
                for j in range(0, n - 1 - i):
                    if key(ar[j]) > key(ar[j + 1]):
                        ar[j], ar[j + 1] = ar[j + 1], ar[j]
            return ar

        idx = [i[0] for i in bubble_sort(list(enumerate(self.stimulator_timing)), key=lambda x:x[1])]

        self.stimulator_timing = [self.stimulator_timing[i] for i in idx]
        self.stimulator_array_list = [self.stimulator_array_list[i] for i in idx]


        # /////////////////////////////////////////////////////////////////////////
        # ///////////////////////// TOGGLE_ARRAY MANIPULATION /////////////////////
        # /////////////////////////////////////////////////////////////////////////

        # Before: toggle_array = [start_t1, end_t1, byte1, bit1, start_t2, ...]
        # After: toggle_array = [event_start_t1, event_end_t1, nb_trains (e.g. 2), byte1, bit1, byte2, bit2, start_t3, ...]

        temp = []
        # make 2-dimensional list
        for i in range(0,int(len(self.toggle_array)/4)):
            temp.extend([self.toggle_array[4*i:4*i+4]])
        self.toggle_array = temp

        copied_array = self.toggle_array.copy()

        stop_events = []
        j = 0
        for i in range(0,len(self.toggle_array)):
            start_event = next((x for x, val in enumerate(self.stimulator_timing) if val == self.toggle_array[i][0]), 0) # the default 0 at the end has to be there (for an unknown reason), otherwise sometimes an error when stimulating multiple electrodes at the exact same time
            stop_event = next(x for x, val in enumerate(self.stimulator_timing) if val > self.toggle_array[i][1]) - 1

            if len(stop_events) > 0 and start_event < stop_events[-1]: # If a toggle has to be added to an already existing toggle
                copied_array[i-1-j][1] = stop_event # Increase the end of the toggle to increase the toggle duration from the start event of the first until the end event of the last
                copied_array[i-1-j].extend([copied_array[i-j][-2]]) # Add the byte array to be toggled
                copied_array[i-1-j].extend([copied_array[i-j][-1]]) # Add the bit to be toggled
                copied_array[i-1-j][2] = copied_array[i-1-j][2] + 1 # Increase the number of electrodes to be toggled
                copied_array.pop(i-j) # Remove the toggle sub-array that has been added to the already existing toggle now
                j = j+1 # j defines how many toggle sub-arrays have been deleted already, so this has to be subtracted from i to get the correct next sub-array
                stop_events.extend([stop_event]) # Add the stop event to the list of stop events to check if the next toggle should be a new operation or should be added to the previously existing toggle

            else: # If a new toggle operation has to be created
                copied_array[i-j][0] = start_event # Add the start event
                copied_array[i-j][1] = stop_event # Add the stop event
                copied_array[i-j].insert(2, 1) # Insert the electrode number (how many electrodes have to be toggled)
                stop_events.extend([stop_event]) # Add the stop event to the list of stop events to check if the next toggle should be a new operation or should be added to the previously existing toggle

        if self.interleaved == True:
            initial_length = len(copied_array)
            for j in range(0, initial_length-1):
                gnd_bytes_and_bits = []
                for gnd_elec in self.gnd_electrodes[j]:
                    switch = gnd_elec // 16
                    connection_byte = (self.nb_levels * 6 - 1 - switch) * 32 + (15 - gnd_elec % 16) // 8
                    gnd_bytes_and_bits.append(connection_byte)
                    connection = gnd_elec % 8
                    gnd_bytes_and_bits.append(connection)
                copied_array.insert(3*j+1, [copied_array[3*j][1]+1, copied_array[3*j+1][0]-1, len(gnd_electrodes[j])])
                copied_array.insert(3*j+2, gnd_bytes_and_bits)

            # For the last stimulation trial
            if len(gnd_electrodes) == 1:
                i = -1
            copied_array.append([copied_array[-1][1]+1, len(self.stimulator_timing)-1, len(gnd_electrodes[j+1])])
            gnd_bytes_and_bits = []
            for gnd_elec in self.gnd_electrodes[j+1]:
                switch = gnd_elec // 16
                connection_byte = (self.nb_levels * 6 - 1 - switch) * 32 + (15 - gnd_elec % 16) // 8
                gnd_bytes_and_bits.append(connection_byte)
                connection = gnd_elec % 8
                gnd_bytes_and_bits.append(connection)
            copied_array.append(gnd_bytes_and_bits)


        self.toggle_array = [item for sublist in copied_array for item in sublist]
        self.nb_toggles = len(self.toggle_array)

        # Gnd array manipulation
        # Note that this has nothing to do with the number of grounds, but for grounding (shorting) during bipolar stimulation.
        for sublist in self.gnd_array:
            # Replace each timing by the last event with the same timing by looking at the first event with a larger timing value, and then subtracting 1
            previous_event = next(x for x, event_time in enumerate(self.stimulator_timing) if event_time > sublist[0]) - 1
            sublist[0] = previous_event

            # Replace each electrode by its switch_byte and switch_bit
            switch = sublist[1] // 16
            connection_byte = (self.nb_levels*6-1-switch)*32 + (15 - sublist[1]%16)//8
            connection = sublist[1]%8
            sublist[1] = 1 # 1 electrode in this sublist
            sublist.append(connection_byte)
            sublist.append(connection)

        self.gnd_array.sort(key=lambda x: x[0]) # Sort in ascending order based on the first element of each sublist ( = the event)

        i = 1
        while i < len(self.gnd_array):
            if self.gnd_array[i][0] == self.gnd_array[i-1][0]:
                sublist = self.gnd_array.pop(i)
                self.gnd_array[i-1].extend(sublist[2:4])
                self.gnd_array[i-1][1] += 1
                i -= 1 # to counteract for the sublist that has been popped 3 lines above this line
            i += 1

        self.gnd_array = [item for sublist in self.gnd_array for item in sublist] # Merge sublists that have the same event
        self.nb_gnds = len(self.gnd_array) # Nothing to do with the number of grounds, but with the number of times you short electrodes in bipolar stimulation

        # Make 1 long bytearray, ready to be transmitted
        for command_array in self.stimulator_array_list:
            self.stimulator_array.extend(command_array)



    def assign_channels(self):
        available_channels = [[i for i in range(0,16)] for _ in range(0,self.nb_levels) ]
        connections = [[],[]]
        for pulse in range(0, len(self.trial_array)):
            electrode = self.trial_array[pulse][0]
            # If electrode is already connected to a channel: connect it again
            if self.trial_array[pulse][0] in connections[0]:
                channel = connections[1][connections[0].index(electrode)]
            # Else take the following available channel
            else:
                lvl = electrode//96
                channel = 16*lvl + available_channels[lvl].pop(0)
            self.trial_array[pulse].insert(1, channel)
            connections[0].append(electrode)
            connections[1].append(channel)


    def write_to_register(self, stim, address, data, timing):
        # possible ways: hexadecimal, integer, or bits as integer (eg int('10110101', base=2))

        command_array = bytearray(self.nb_stimulator_bytes)  # last 3 bytes are for stim 0, etc
        command_array[(self.nb_stimulator_bytes - 3 - stim * 3)] = address
        command_array[(self.nb_stimulator_bytes - 3 - stim * 3) + 1] = ((data >> 8) & 0xFF)
        command_array[(self.nb_stimulator_bytes - 3 - stim * 3) + 2] = (data & 0xFF)

        # print(command_array.hex())
        # print(self.show_bits(command_array))

        self.stimulator_array_list.extend([command_array])
        self.stimulator_timing.extend([timing])

    # helper functions to check the bits
    def show_bits(self, data):
        return [self.access_bit(data,i) for i in range(len(data)*8)]

    def access_bit(self, data, num):
        base = int(num // 8)
        shift = int(num % 8)
        return (data[base] & (1 << shift)) >> shift



# ====================
# ====================
# ====================
# ====================
# ====================


### Following function not needed for correct functioning of trial --> no need to implement this
def plot_adc(adc_lst):
    if adc_lst != None:
        import matplotlib.pyplot as plt

        # reject outliers (hardware mistakes in ADC sometimes happen)
        #adc_lst = [adc_lst[i] for i in range(1,len(adc_lst)-1) if (min(abs(adc_lst[i-1]),abs(adc_lst[i+1]))-1 < adc_lst[i] < max(abs(adc_lst[i-1]),abs(adc_lst[i+1]))+1) ]
        adc_lst = [e for e in adc_lst if e != -12]

        logging.disable()
        #print(adc_lst)

        plt.plot( adc_lst, 'b-', adc_lst, 'b.', MarkerSize=2)
        '''
        plt.ylim([-12,12])
        plt.ylabel('Voltage [V]')
        plt.xticks(range(0,len(adc_lst)+1,25),[str(i*4) for i in range(0,len(adc_lst)+1,25)])
        plt.xlabel('Time [us]')
        '''
        plt.show()

def visualize_stimulation(trial_array):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    x_values = [0]
    y_values = [0]

    # Create waveform per pulse
    colors = cm.rainbow(np.linspace(0,1,len(set([i[0] for i in trial_array]))))
    i = 0
    used_channels = [trial_array[0][0]] # add first channel already

    plt.close('all')

    for pulse in range(0,len(trial_array)):
        x_values.append(trial_array[pulse][3])
        y_values.append(0)

        x_values.append(trial_array[pulse][3])
        y_values.append(trial_array[pulse][2])

        x_values.append(trial_array[pulse][3]+trial_array[pulse][4])
        y_values.append(trial_array[pulse][2])

        x_values.append(trial_array[pulse][3]+trial_array[pulse][4])
        y_values.append(0)

        x_values.append(trial_array[pulse][6])
        y_values.append(0)

        x_values.append(trial_array[pulse][6])
        y_values.append(trial_array[pulse][5])

        x_values.append(trial_array[pulse][6]+trial_array[pulse][7])
        y_values.append(trial_array[pulse][5])

        x_values.append(trial_array[pulse][6]+trial_array[pulse][7])
        y_values.append(0)

        plt.plot(x_values, y_values, color=colors[i])

        # Give every channel a different color
        if pulse < len(trial_array)-1:
            if trial_array[pulse+1][0] not in used_channels:
                used_channels.append(trial_array[pulse+1][0])
                i += 1
                x_values = [min(trial_array[pulse+1][3],trial_array[pulse][6]+trial_array[pulse][7])]
                y_values = [0]


    plt.show()







if __name__ == '__main__':
    debug_on_screen = True
    if debug_on_screen:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(module)s %(message)s'))
        # logging.fc (debug, warn, error, info, etc)
        handler.setLevel(logging.INFO) # set at DEBUG if you want more information
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO) # set at DEBUG if you want more information

###############################################
### ONLY CHANGE OR ADD CODE BELOW THIS LINE ###
###############################################

    USB_port = 'COM8'                   # Set the correct name of the USB_port
    trial = TransmitTrial(USB_port)     # Initializes the code, do not change
    bip_dict_list = []                  # Initializes the dictionary, do not change

    '''
    ### Example 1: stimulation trial with 2 pulse trains on 2 different electrodes.

    # Define stimulation trial. All parameters below can be changed
        # First pulse train (40 pulses with a 200 Hz frequency) of the trial, on electrode 14, cathodic-first with -20 µA and 200 µs pulse width.
    bip_dict_list.append(
        {'nb_pulses': 40, 'start': 0, 'amplitude': -20, 'duration': 200, 'electrode': 5, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        # Second pulse train (25 pulses with a 100 Hz frequency) of the trial, on electrode 14, anodic-first with 30 µA and 150 µs pulse width, starts 300 000 µs after the start of the first pulse train
    bip_dict_list.append(
        {'nb_pulses': 25, 'start': 300000, 'amplitude': 30, 'duration': 150, 'electrode': 14, 'frequency': 100, 'interpulse interval': 0, 'multiplier': 1})
        # Third pulse train (30 pulses with a 150 Hz frequency) of the trial, on electrode 16, anodic-first with 40 µA and 170 µs pulse width, starts 700 000 µs after the start of the first pulse train
    bip_dict_list.append(
        {'nb_pulses': 25, 'start': 700000, 'amplitude': 30, 'duration': 150, 'electrode': 16, 'frequency': 100, 'interpulse interval': 0, 'multiplier': 1})

    # Define a list of ground electrodes (e.g. all electrodes on a needle, only 1 electrode or even an empty list is also a possibility).
    gnd_electrodes = [9, 11]

    nb_repeats = 35                         # Number of times the above defined stimulation trial will be identically repeated
    trial_delay = 7800                      # Delay between different stimulation trials, in ms. This is the actual waiting period between the last pulse from the previous trial and the first pulse from the next trial.
    trial_configuration = ConfigureTrial.from_bipolar_train(bip_dict_list, gnd_electrodes=gnd_electrodes, nb_repeats = nb_repeats, trial_delay = trial_delay) # Processes the stimulation parameters, do not change
    trial.stimulation(trial_configuration)  # Starts the stimulation, do not change
    '''

    '''
    ### Example 2: more complex example on a possible stimulation pattern, as illustration (dynamic current steering)
    bip_dict_list = []
    total_amp = -20  # uA
    ratio = 1
    elec1 = 9
    elec2 = 24
    
    bip_dict_list.append({'nb_pulses': 100, 'start': 0, 'amplitude': total_amp*ratio, 'duration': 200, 'electrode': elec1, 'frequency': 100, 'interpulse interval': 0, 'multiplier': 1})
    bip_dict_list.append({'nb_pulses': 100, 'start': 0, 'amplitude': total_amp*(1-ratio), 'duration': 200, 'electrode': elec2, 'frequency': 100, 'interpulse interval': 0, 'multiplier': 1})
    trial_configuration = ConfigureTrial.from_bipolar_train(bip_dict_list, gnd_electrodes=[11, 13, 15], nb_repeats = 5, trial_delay = 2000)
    trial.stimulation(trial_configuration)



    '''

    '''
    ### Example 2: more complex example on a possible stimulation pattern, as illustration (dynamic current steering)
    bip_dict_list = []
    total_amp = 20  # uA
    elec1 = 19
    elec2 = 17
    
    bip_dict_list.append({'nb_pulses': 20, 'start': 0, 'amplitude': total_amp, 'duration': 200, 'electrode': elec1, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
    bip_dict_list.append({'nb_pulses': 10, 'start': 100000, 'amplitude': total_amp-total_amp/3, 'duration': 200, 'electrode': elec1, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
    bip_dict_list.append({'nb_pulses': 10, 'start': 100000, 'amplitude': total_amp/3, 'duration': 200, 'electrode': elec2, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
    bip_dict_list.append({'nb_pulses': 10, 'start': 150000, 'amplitude': total_amp-total_amp/3*2, 'duration': 200, 'electrode': elec1, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
    bip_dict_list.append({'nb_pulses': 10, 'start': 150000, 'amplitude': total_amp/3*2, 'duration': 200, 'electrode': elec2, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
    bip_dict_list.append({'nb_pulses': 20, 'start': 200000, 'amplitude': total_amp, 'duration': 200, 'electrode': elec2, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})

    trial_configuration = ConfigureTrial.from_bipolar_train(bip_dict_list, gnd_electrodes=[11, 13, 15], nb_repeats = 35, trial_delay = 7800)
    trial.stimulation(trial_configuration)
    '''

    ### EXAMPLE - STIMULATION PROTOCOL USED BY MAARTEN
    central_electrode = 32
    return_electrode = 20

    experiment = 1              # 1-CENTRAL ELEC, 5-RETURN ELEC, 2-BIPOLAR
    experiment2_subset = 1      # 1 (3900 frames), 2 (3180 frames), or 3 (3180 frames)

    if experiment == 1:
        ### EXPERIMENT 1: CONVENTIONAL STIMULATION --- NOTHING TO BE CHANGED
        print("Execute experiment 1")
        #10,15,20,25,30 uA CENTRAL ELECTRODES
        bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': 10, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        #bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': 20, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        #bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': 30, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        #bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': 15, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        #bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': 25, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})

        #central_electrode = 19
        #gnd_electrodes = [_ for _ in range(19, 32) if _ not in [19, 31, 21, 29, 23, 27, 25]] # All electrodes except for the ones on the same needle as the central electrode
        # central_electrode = 32
        gnd_electrodes = [_ for _ in range(19, 32) if _ not in [32, 20, 30, 22, 28, 24, 26]]
        trial_configuration = ConfigureTrial.from_bipolar_train(bip_dict_list, gnd_electrodes=gnd_electrodes, nb_repeats=1, trial_delay=5000, interleaved=False)

    elif experiment == 5:
        # 10,15,20,25,30 uA  RETURN ELECTRODE 31
        bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': 10, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': 20, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': 30, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': 15, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})
        bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': 25, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})

        # return_electrode = 31
        gnd_electrodes = [_ for _ in range(19, 32) if _ not in [19, 31, 21, 29, 23, 27,25]]  # All electrodes except for the ones on the same needle as the central electrode
        trial_configuration = ConfigureTrial.from_bipolar_train(bip_dict_list, gnd_electrodes=gnd_electrodes,nb_repeats=10, trial_delay=5000, interleaved=False)

    elif experiment == 2:
        ### EXPERIMENT 2: DIRECTIONALITY --- ONLY CHANGE AMPLITUDE

        if experiment2_subset == 1:
            #10, 15, 20, 25, 30 uA
            print("Execute experiment 2")
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': -10, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': 10, 'duration': 200, 'electrode': return_electrode, 'frequency': 200,'interpulse interval': 0, 'multiplier': 1})  # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': -20, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': 20, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -30, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': 30, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': -15, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': 15, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': -25, 'duration': 200, 'electrode': central_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': 25, 'duration': 200, 'electrode': return_electrode,'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

        elif experiment2_subset == 2:
            print("Execute experiment 2 - subset B - amplitude " + str(experiment2_amp))
            # Part B
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': amp/2, 'duration': 200, 'electrode': return6, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': amp/2, 'duration': 200, 'electrode': return7, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': amp, 'duration': 200, 'electrode': return3, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': amp, 'duration': 200, 'electrode': return6, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': amp/2, 'duration': 200, 'electrode': return4, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': amp/2, 'duration': 200, 'electrode': return5, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

        elif experiment2_subset == 2:
            print("Execute experiment 2 - subset C - amplitude " + str(experiment2_amp))
            # Part C
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': amp, 'duration': 200, 'electrode': return4, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1})  # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': amp, 'duration': 200, 'electrode': return2, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': amp/2, 'duration': 200, 'electrode': return5, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # anodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': amp/2, 'duration': 200, 'electrode': return6, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': amp, 'duration': 200, 'electrode': return7, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # anodic

        gnd_electrodes = []
        trial_configuration = ConfigureTrial.from_bipolar_train(bip_dict_list, gnd_electrodes=gnd_electrodes, nb_repeats=10, trial_delay=5000, interleaved=False)

    elif experiment == 3:
        ### EXPERIMENT 3: LOCALIZATION
        central_electrode = central_electrode
        return_column = 7
        return_layer = 42
        return_diagonal = 9
        return_same_needle = 40
        electrode_next_needle = return_layer
        return_next_needle_same_needle = 52
        return_next_needle_low = 46
        return_central_needle_low = 35

        multiplier = experiment3_multiplier

        if experiment3_subset == 1:
            # Part A: subset 1 and cathodic-first, amps 10 and 20, already been done with multiplier 1, still to be done with multiplier 4
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': -10, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': 10, 'duration': 200, 'electrode': return_column, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': -10, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': 10, 'duration': 200, 'electrode': return_layer, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -10, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': 10/2, 'duration': 200, 'electrode': return_column, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': 10/2, 'duration': 200, 'electrode': return_diagonal, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': -20, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': 20, 'duration': 200, 'electrode': return_column, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': -20, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': 20, 'duration': 200, 'electrode': return_layer, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': -20, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': 20/2, 'duration': 200, 'electrode': return_column, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': 20/2, 'duration': 200, 'electrode': return_diagonal, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

        elif experiment3_subset == 2:
            # Part B: subset 1 and anodic-first, amps 10 and 20, to be done with multiplier 1 and 4
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': 10, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': -10, 'duration': 200, 'electrode': return_column, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': 10, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': -10, 'duration': 200, 'electrode': return_layer, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': 10, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -10/2, 'duration': 200, 'electrode': return_column, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -10/2, 'duration': 200, 'electrode': return_diagonal, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': 20, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': -20, 'duration': 200, 'electrode': return_column, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': 20, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': -20, 'duration': 200, 'electrode': return_layer, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': 20, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': -20/2, 'duration': 200, 'electrode': return_column, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': -20/2, 'duration': 200, 'electrode': return_diagonal, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

        elif experiment3_subset == 3:
            # Part C: subset 2 (return on same needles), cathodic-first and anodic-first, amps 10 and 20, and multiplier 1 and 4
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': -10, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': 10, 'duration': 200, 'electrode': return_same_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': -10, 'duration': 200, 'electrode': electrode_next_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': 10, 'duration': 200, 'electrode': return_next_needle_same_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': 10, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -10, 'duration': 200, 'electrode': return_same_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': 10, 'duration': 200, 'electrode': electrode_next_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': -10, 'duration': 200, 'electrode': return_next_needle_same_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': -20, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': 20, 'duration': 200, 'electrode': return_same_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': -20, 'duration': 200, 'electrode': electrode_next_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': 20, 'duration': 200, 'electrode': return_next_needle_same_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 48000000, 'amplitude': 20, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 48000000, 'amplitude': -20, 'duration': 200, 'electrode': return_same_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 56000000, 'amplitude': 20, 'duration': 200, 'electrode': electrode_next_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 56000000, 'amplitude': -20, 'duration': 200, 'electrode': return_next_needle_same_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

        elif experiment3_subset == 4:
            # Part D: subset 3 (includes lower layers), cathodic-first and anodic-first, to be done with amps 10 and 20, and multiplier 1 and 4
            multiplier = experiment3_multiplier
            amp = experiment3_amp

            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': -amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': amp, 'duration': 200, 'electrode': return_next_needle_low, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_next_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': amp, 'duration': 200, 'electrode': return_central_needle_low, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -amp, 'duration': 200, 'electrode': return_central_needle_low, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': amp, 'duration': 200, 'electrode': return_next_needle_low, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': amp, 'duration': 200, 'electrode': central_electrode, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': -amp, 'duration': 200, 'electrode': return_next_needle_low, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': amp, 'duration': 200, 'electrode': electrode_next_needle, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': -amp, 'duration': 200, 'electrode': return_central_needle_low, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': amp, 'duration': 200, 'electrode': return_central_needle_low, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # cathodic
            bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': -amp, 'duration': 200, 'electrode': return_next_needle_low, 'frequency': 200, 'interpulse interval': 0, 'multiplier': multiplier}) # anodic

        gnd_electrodes = []
        trial_configuration = ConfigureTrial.from_bipolar_train(bip_dict_list, gnd_electrodes=gnd_electrodes, nb_repeats=10, trial_delay=7800, interleaved=False)

    elif experiment == 4:
        ### EXPERIMENT 4: THRESHOLD --- NOTHING TO BE CHANGED (only the return electrode if necessary)
        # amp = 10
        # real_amp = int((amp+24000)*65536/48000)/65536*48000-24000 # Taking the 1.4uA resolution into account
        # print(real_amp)
        electrode_A = central_electrode
        electrode_B = 41

        amp = 7 # actually 6.6uA
        bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_A, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # cathodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 0, 'amplitude': amp, 'duration': 200, 'electrode': electrode_B, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # anodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_A, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 4})  # cathodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 8000000, 'amplitude': amp, 'duration': 200, 'electrode': electrode_B, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 4})  # anodic

        amp = 10 # actually 9.5uA

        amp = 13 # actually 12.5
        bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_A, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # cathodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 16000000, 'amplitude': amp, 'duration': 200, 'electrode': electrode_B, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # anodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_A, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 4})  # cathodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 24000000, 'amplitude': amp, 'duration': 200, 'electrode': electrode_B, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 4})  # anodic

        amp = 16 # actually 15.4
        bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_A, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # cathodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 32000000, 'amplitude': amp, 'duration': 200, 'electrode': electrode_B, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # anodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_A, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 4})  # cathodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 40000000, 'amplitude': amp, 'duration': 200, 'electrode': electrode_B, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 4})  # anodic

        amp = 20 # actually 19.8uA

        amp = 23 # actually 22.7
        bip_dict_list.append({'nb_pulses': 40, 'start': 48000000, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_A, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # cathodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 48000000, 'amplitude': amp, 'duration': 200, 'electrode': electrode_B, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 1}) # anodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 56000000, 'amplitude': -amp, 'duration': 200, 'electrode': electrode_A, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 4})  # cathodic
        bip_dict_list.append({'nb_pulses': 40, 'start': 56000000, 'amplitude': amp, 'duration': 200, 'electrode': electrode_B, 'frequency': 200, 'interpulse interval': 0, 'multiplier': 4})  # anodic

        gnd_electrodes = []
        trial_configuration = ConfigureTrial.from_bipolar_train(bip_dict_list, gnd_electrodes=gnd_electrodes, nb_repeats=10, trial_delay=7800, interleaved=False)

    trial.stimulation(trial_configuration)  # Starts the stimulation, do not change