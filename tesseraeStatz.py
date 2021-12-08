"""
this will do stats on balloon paths Lebonnois sent via personal communication and can plot one of the 84 paths, takes like an hour to run on my aged comp
balloon path (balloon2021.tgz - could use balloon2017.tgz data but Lebonnois said 2021 latest&greatest (may not have a publication on it yet)) from personal communication w/ Lebonnois
if you look at the .out files I think 'Zare' is the altitude (ignore Zsurf b/c this code makes its own time & longitude based on the days/circumnav req - it essentially uses the different starting longitudes as an analogy for landing at different times of the day)
tesserae map from Gilmore et al., 2020 [paper in HW->cited], edited to remove dots but dot copy still there
tesserae outlines digitized from Gilmore et al., 2020 tesserae map, if you want to add more they can only have 4 corners to work with the alg
this revolves around "major tesserae regions" which are the extra # at the end of a tesserae box coord list
0 is no major region (little bits out in the middle of nowhere) and 1-13 are major regions I identified as worthy entirely based on looking at them and being like "yo that's a big region"
I visualized each tbox one by one by plotting just the figure start + figure map + then iterating through the tboxes (there's a i += 1 w/ comment about that)
licensed under MIT (e.g. yolo license) (note subfun_figFitter is GPL-3.0 via previous https://github.com/dinsmoro/GRITI release)
"""
import numpy as np
from scipy import interpolate
import scipy.stats
import pandas as pd
import os
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    from subfun_figFitter import figFitter
except:
    pass;
#END TRY
#--- inputs ---
days = 30; #days to check
obsGoal = 4; #number of observations needed to reach goal
obsGoal_sep = 2; #number of seperate tesserae regions needed to reach goal
daysPerCircum = 4.5; #days per circumnavigation [6.8 via estimate math, 4.5 via people]
landingCoords = np.array((100+180,0)); #long, lat for West Ovda optimal insertion point
dataRate = 10; #[datapts/sec; Hz] sampling rate
ptsInTesseraeReq = 10; #pts needed to make a tesserae measurement count
dataFolder = '/data/Lat-1010_p2e4/'; #note the data folder
goodLatitudes = np.array([-10,10]); #good latitude range [is either -10,-7,  -4, 0, 4, 7, 10]
#plot stuff
plot_latLim = 90; #+/- latitude limit to plot to
plot_latSpacing = 15; #spacing between latitude axis pts
plot_toGoalOnly = False; #plots line to goal, green check at end
toGoal = [4,6,6]; #the goals to go to (special from obsGoal!)
#cache
cacheItUp = False; #DO NOT TURN ON HERE

#--- Calcs from Above ---
dataPtsPerCircum = np.int64(np.floor(daysPerCircum*86400*dataRate)); #data pts per circum
circums = days/daysPerCircum; #circumnavigations
longPerCircum = 360; #[long/circum]

#--- build platform path for latitude/days ---
platformPaths = []; #make a dict
filez = os.listdir(os.getcwd()+dataFolder);
for i in range(0,len(filez)):
    if( filez[i].rfind('.out') > 0 ):
        headerText = ['Time(Edays)','Latitude','Longitude','LT (Vh)','Press (Pa)','Temp (K)','Density(m-3)','U (m/s)','V (m/s)','Zare (km)','Zsurf (km)'];
        gg = pd.read_csv(os.getcwd()+dataFolder+filez[i], delim_whitespace=True,header=None,skiprows=1).to_numpy();
        if( (gg[0,1] <= np.max(goodLatitudes)) & (gg[0,1] >= np.min(goodLatitudes)) ):
            #if data within good lat range
            platformPaths.append(gg[:,0:2]); #only care about time/lat in this really simple deal
        #END IF
    #END IF
#END FOR i

#--- build tesserae box list ---
tbox = []; #[X: longitude, Y: latitude] boxes from Gilmore et al. 2020 tesserae plot roughly
tbox.append([np.array(((-2.228335625859671, -27.092783505154642),
(7.9229711141678365, -32.041237113402076),
(12.379642365887236, -21.154639175257728),
(1.9807427785419804, -17.44329896907216))),2]);
tbox.append([np.array(((55.21320495185694, -15.216494845360828),
(104.97936726272354, -11.257731958762875),
(106.21733149931225, 5.567010309278345),
(58.6795048143054, 5.814432989690729))),1]);
tbox.append([np.array(((114.63548830811555, -4.824742268041234),
(139.39477303988997, -3.0927835051546424),
(136.67125171939477, -17.69072164948453),
(119.58734525447045, -11.75257731958763))),5]);
tbox.append([np.array(((111.911967,	-0.618556701),
(114.3878955,	8.783505155),
(124.786795,	8.536082474),
(119.8349381,	-1.608247423))),5]);
tbox.append([np.array(((127.7579092,	2.350515464),
(127.0151307,	9.773195876),
(139.394773,	6.309278351),
(140.1375516,	0.12371134))),5]);
tbox.append([np.array(((153.2599725,	-9.030927835),
(157.9642366,	-4.577319588),
(163.1636864,	-7.793814433),
(162.4209078,	-11.75257732))),0]);
tbox.append([np.array(((119.0921596,	10.26804124),
(122.3108666,	24.37113402),
(125.2819807,	22.88659794),
(122.5584594,	10.7628866))),5]);
tbox.append([np.array(((112.1595598,	-35.75257732),
(116.8638239,	-33.03092784),
(117.8541953,	-34.51546392),
(113.8927098,	-37.48453608))),0]);
tbox.append([np.array(((62.64099037,	-23.3814433),
(78.23933975,	-18.68041237),
(78.4869326,	-22.39175258),
(64.869326,	-25.60824742))),0]);
tbox.append([np.array(((51.49931224,	-5.072164948),
(50.26134801,	-0.618556701),
(48.03301238,	-1.113402062),
(48.04301238,	-5.072164948))),0]);
tbox.append([np.array(((38.62448418,	-4.082474227),
(43.08115543,	1.113402062),
(38.62448418,	3.587628866),
(36.6437414,	-2.597938144))),0]);
tbox.append([np.array(((45.30949106,	2.845360825),
(49.76616231,	2.845360825),
(48.77579092,	0.618556701),
(46.0522696,	1.113402062))),0]);
tbox.append([np.array(((42.09078404,	-6.804123711),
(46.0522696,	-7.793814433),
(44.56671252,	-9.525773196),
(41.3480055,	-10.02061856))),0]);
tbox.append([np.array(((62.64099037,	-32.04123711),
(39.36726272,	-31.05154639),
(37.63411279,	-34.26804124),
(59.42228336,	-35.50515464))),0]);
tbox.append([np.array(((61.65061898,	-38.72164948),
(47.53782669,	-37.73195876),
(48.03301238,	-39.95876289),
(57.44154058,	-41.69072165))),0]);
tbox.append([np.array(((9.903713893,	-48.37113402),
(16.09353508,	-48.37113402),
(17.82668501,	-53.56701031),
(10.64649243,	-53.81443299))),6]);
tbox.append([np.array(((21.54057772,	-55.05154639),
(30.94910591,	-58.51546392),
(18.56946355,	-70.1443299),
(8.418156809,	-65.44329897))),6]);
tbox.append([np.array(((103.741403,	-49.36082474),
(106.2173315,	-47.87628866),
(107.4552957,	-48.86597938),
(105.2269601,	-50.59793814))),0]);
tbox.append([np.array(((97.79917469,	-28.57731959),
(102.0082531,	-23.62886598),
(104.9793673,	-26.59793814),
(100.2751032,	-30.55670103))),0]);
tbox.append([np.array(((106.2173315,	-28.82474227),
(108.4456671,	-26.35051546),
(109.6836314,	-27.09278351),
(108.1980743,	-28.82474227))),0]);
tbox.append([np.array(((29.46354883,	-38.96907216),
(30.94910591,	-43.91752577),
(31.9394773,	-43.17525773),
(30.45392022,	-38.4742268))),0]);
tbox.append([np.array(((126.519945,	26.35051546),
(129.4910591,	33.03092784),
(126.2723521,	34.02061856),
(122.0632737,	30.55670103))),0]);
tbox.append([np.array(((109.4360385,	43.42268041),
(114.6354883,	38.22680412),
(109.4360385,	36.98969072),
(106.96011,	40.70103093))),0]);
tbox.append([np.array(((112.6547455,	36.98969072),
(117.3590096,	32.28865979),
(118.349381,	33.77319588),
(113.8927098,	37.97938144))),0]);
tbox.append([np.array(((113.1499312,	23.3814433),
(111.911967,	25.36082474),
(110.9215956,	23.62886598),
(111.911967,	23.13402062))),0]);
tbox.append([np.array(((104.7317744,	35.01030928),
(102.5034388,	36.98969072),
(97.303989,	33.03092784),
(98.78954608,	30.06185567))),0]);
tbox.append([np.array(((93.09491059,	26.84536082),
(98.78954608,	30.06185567),
(98.04676754,	31.05154639),
(92.84731774,	27.58762887))),0]);
tbox.append([np.array(((94.58046768,	30.55670103),
(99.03713893,	32.28865979),
(98.29436039,	33.77319588),
(94.33287483,	31.54639175))),0]);
tbox.append([np.array(((83.43878955,	26.59793814),
(94.82806052,	46.88659794),
(73.53507565,	42.68041237),
(75.76341128,	28.08247423))),4]);
tbox.append([np.array(((59.91746905,	25.36082474),
(60.90784044,	26.10309278),
(59.6698762,	26.59793814),
(58.67950481,	25.60824742))),0]);
tbox.append([np.array(((56.94635488,	28.57731959),
(59.17469051,	32.04123711),
(56.20357634,	32.28865979),
(54.47042641,	28.32989691))),0]);
tbox.append([np.array(((55.4607978,	23.13402062),
(56.20357634,	25.11340206),
(54.47042641,	25.36082474),
(54.47042641,	23.62886598))),0]);
tbox.append([np.array(((45.80467675,	19.91752577),
(45.06189821,	20.65979381),
(47.53782669,	27.58762887),
(48.28060523,	27.09278351))),0]);
tbox.append([np.array(((39.11966988,	24.12371134),
(41.8431912,	27.58762887),
(40.60522696,	28.57731959),
(38.37689133,	24.86597938))),0]);
tbox.append([np.array(((45.06189821,	31.54639175),
(44.56671252,	32.28865979),
(46.5474553	,34.02061856),
(47.29023384,	33.27835052))),0]);
tbox.append([np.array(((32.434663,	30.55670103),
(30.20632737,	31.29896907),
(31.9394773	,35.25773196),
(32.92984869,	33.03092784))),0]);
tbox.append([np.array(((28.47317744,	34.51546392),
(28.96836314,	36.74226804),
(25.74965612,	39.71134021),
(24.75928473,	37.2371134))),0]);
tbox.append([np.array(((38.87207703,	5.319587629),
(38.12929849,	5.319587629),
(39.86244842,	11.01030928),
(40.60522696,	11.01030928))),0]);
tbox.append([np.array(((1.733149931,	33.03092784),
(-0.742778542,	34.7628866),
(1.980742779,	37.73195876),
(2.72352132,	36.24742268))),0]);
tbox.append([np.array(((-1.237964237,	40.20618557),
(-3.466299862,	43.42268041),
(-1.980742779,	43.67010309),
(-0.247592847,	41.93814433))),0]);
tbox.append([np.array(((-62.88858322,	-11.25773196),
(-58.43191197,	-3.340206186),
(-61.89821183,	-0.12371134),
(-68.08803301,	-7.298969072))),7]);
tbox.append([np.array(((-65.11691884,	-0.371134021),
(-66.60247593,	4.082474227),
(-64.869326,	4.577319588),
(-63.38376891,	0.618556701))),7]);
tbox.append([np.array(((-67.34525447,	7.051546392),
(-65.36451169,	11.75257732),
(-62.64099037,	8.536082474),
(-64.62173315,	7.051546392))),7]);
tbox.append([np.array(((-69.5735901,	16.45360825),
(-62.88858322,	16.20618557),
(-61.15543329,	28.32989691),
(-66.85006878,	30.30927835))),8]);
tbox.append([np.array(((-58.43191197,	28.82474227),
(-56.45116919,	33.77319588),
(-60.90784044,	36),
(-63.38376891,	31.29896907))),8]);
tbox.append([np.array(((-70.06877579,	29.81443299),
(-69.82118294,	36.74226804),
(-85.17193948,	36.24742268),
(-90.12379642,	32.78350515))),8]);
tbox.append([np.array(((-94.08528198,	25.8556701),
(-89.62861073,	29.07216495),
(-92.1045392,	31.05154639),
(-96.31361761,	28.08247423))),0]);
tbox.append([np.array(((-96.06602476,	29.56701031),
(-97.05639615,	31.79381443),
(-114.3878955,	29.81443299),
(-109.4360385,	27.58762887))),0]);
tbox.append([np.array(((-98.78954608,	33.5257732),
(-97.05639615,	37.2371134),
(-100.0275103,	37.48453608),
(-100.7702889,	34.26804124))),0]);
tbox.append([np.array(((-73.28748281,	-14.96907216),
(-76.50618982,	-3.340206186),
(-82.94360385,	-2.350515464),
(-79.47730399,	-15.46391753))),9]);
tbox.append([np.array(((-83.93397524,	-4.082474227),
(-84.18156809,	-1.113402062),
(-93.59009629,	-1.113402062),
(-92.84731774,	-4.329896907))),9]);
tbox.append([np.array(((-93.59009629,	-4.577319588),
(-95.81843191,	-3.835051546),
(-97.79917469,	-8.041237113),
(-96.06602476,	-8.288659794))),9]);
tbox.append([np.array(((-108.69326,	19.91752577),
(-106.96011,	21.15463918),
(-109.9312242,	23.87628866),
(-110.6740028,	22.1443299))),0]);
tbox.append([np.array(((-162.9160935,	24.6185567),
(-162.6685007,	27.09278351),
(-167.1251719,	29.31958763),
(-167.6203576,	27.09278351))),0]);
tbox.append([np.array(((-171.3342503,	16.94845361),
(-170.8390646,	17.69072165),
(-172.8198074,	18.43298969),
(-173.8101788,	17.19587629))),0]);
tbox.append([np.array(((-171.5818432,	27.58762887),
(-168.3631362,	32.28865979),
(-172.0770289,	33.27835052),
(-174.3053645,	28.82474227))),0]);
tbox.append([np.array(((-178.2668501,	31.29896907),
(-178.2668501,	33.27835052),
(-180,	34.51546392),
(-180,	32.28865979))),0]);
tbox.append([np.array(((169.8486933,	43.17525773),
(174.0577717,	42.43298969),
(173.562586,	40.94845361),
(170.5914718,	41.69072165))),0]);
tbox.append([np.array(((168.8583219,	41.44329897),
(170.8390646,	40.20618557),
(170.0962861,	38.72164948),
(169.1059147,	39.21649485))),0]);
tbox.append([np.array(((171.829436,	39.21649485),
(181.9807428,	38.4742268),
(181.9807428,	35.25773196),
(172.3246217,	37.48453608))),0]);
tbox.append([np.array(((181.9807428,	34.7628866),
(178.0192572,	36),
(178.5144429,	33.5257732),
(180.4951857,	33.03092784))),0]);
tbox.append([np.array(((-67.09766162,	-26.35051546),
(-64.869326,	-26.35051546),
(-64.3741403,	-28.57731959),
(-67.09766162,	-28.57731959))),0]);
tbox.append([np.array(((27.3983447283195, 47.01881331403763),
(52.78877293990638, 42.06946454413893),
(62.11586901763221, 54.57308248914617),
(46.31162288593015, 58.74095513748191))),10]);
tbox.append([np.array(((93.46527527887721, 64.73227206946454),
(124.29650953580418, 65.5137481910275),
(123.51925152932702, 69.16063675832127),
(97.092479309104, 67.07670043415341))),11]);
tbox.append([np.array(((9.521410579345059, 53.27062228654125),
(83.87909319899236, 68.37916063675833),
(89.5789852464915, 74.63096960926194),
(13.148614609571752, 75.67293777134587))),3]);
tbox.append([np.array(((-27.527887729399083, 53.27062228654125),
(-15.350845627923746, 57.17800289435601),
(-15.86901763224185, 61.866859623733724),
(-26.232457718603825, 59.0014471780029))),0]);
tbox.append([np.array(((-58.10003598416699, 60.82489146164979),
(-50.586541921554556, 56.65701881331404),
(-45.1457358762145, 60.04341534008683),
(-53.695573947463146, 62.12735166425471))),0]);
tbox.append([np.array(((-16.1281036344009, 70.20260492040521),
(-0.8420295070169459, 68.63965267727932),
(-0.0647715005397913, 72.54703328509407),
(-14.832673623605643, 77.23589001447178))),0]);
tbox.append([np.array(((-49.2911119107593, 75.93342981186686),
(-17.16444764303708, 73.589001447178),
(-15.350845627923746, 76.71490593342982),
(-45.66390788053258, 78.5383502170767))),0]);
tbox.append([np.array(((-72.09068010075569, 72.80752532561506),
(-58.10003598416699, 71.24457308248915),
(-57.06369197553079, 76.19392185238785),
(-67.68621806405183, 76.45441389290883))),0]);
tbox.append([np.array(((-130.90320259086002, 56.65701881331404),
(-120.28067650233899, 50.92619392185239),
(-112.76718243972654, 62.908827785817664),
(-131.93954659949623, 62.38784370477569))),12]);
tbox.append([np.array(((-163.8071248650594, 53.531114327062234),
(-160.69809283915077, 52.74963820549928),
(-158.36631881971934, 67.59768451519537),
(-161.73443684778698, 68.90014471780029))),0]);
tbox.append([np.array(((67.55667506297223, 45.45586107091173),
(72.47930910399421, 45.45586107091173),
(73.25656711047137, 47.27930535455861),
(71.442965095358, 49.1027496382055))),4]);
tbox.append([np.array(((75.84742713206188, 45.97684515195369),
(82.32457718603806, 47.01881331403763),
(81.02914717524285, 50.14471780028944),
(74.29291111910757, 50.14471780028944))),4]);
tbox.append([np.array(((85.17452320978768, 46.237337192474676),
(92.16984526808201, 46.49782923299566),
(86.21086721842383, 52.74963820549928),
(83.36092119467432, 51.44717800289436))),4]);
tbox.append([np.array(((81.2882331774019, 52.74963820549928),
(83.10183519251521, 55.35455861070912),
(78.69737315581139, 57.95947901591896),
(77.14285714285708, 57.43849493487699))),4]);
tbox.append([np.array(((63.670385030586516, 55.6150506512301),
(74.81108312342568, 56.91751085383502),
(75.58834112990283, 59.0014471780029),
(66.77941705649508, 58.74095513748191))),4]);
tbox.append([np.array(((124.55559553796328, 49.36324167872649),
(129.4782295789852, 45.195369030390744),
(145.80064771500534, 55.09406657018814),
(140.61892767182437, 57.95947901591896))),13]);

#convert tbox from -180 to 180 to 0 to 360
for i in range(0,len(tbox)):
    tbox[i][0][:,0] += 180; #in place increment
#END FOR i
for i in range(0,len(tbox)):
    tbox[i][0][tbox[i][0][:,0] > 360,0] = 360; #keep right
#END FOR i

locz_dict = []; #prep a dict
obz_dict = {}; #prep a dict
obz_dict['obz'] = []; #prep sublist
obz_dict['obz_inTot'] = []; #prep sublist
obz_dict['obz_in'] = []; #prep sublist
obz_dict['obz_inOrdered'] = []; #prep sublist
obz_dict['obz_inOrdered_zones'] = []; #prep sublist
locz = np.empty((np.int64(circums*dataPtsPerCircum)+1,3)); #preallocate [long, lat, time days]
locz[:,2] = np.arange(0,days*86400+1/dataRate,1/dataRate)/86400; #[days]
locz[:,0] = np.mod(-locz[:,2]/daysPerCircum*longPerCircum+landingCoords[0],360); #calc longitudes
for k in range(0,len(platformPaths)):
    #--- build path interpolator ---
    platformPath_raw = platformPaths[k]; #get the path needed
    platformPath_interper = interpolate.UnivariateSpline(platformPath_raw[:,0],platformPath_raw[:,1]); #use this to input any day portion and get out latitudes
    
    #--- build a big array of everything ---
    locz[:,1] = platformPath_interper(locz[:,2]); #calc the latitudes
    obz = np.zeros((np.int64(circums*dataPtsPerCircum)+1),dtype=np.bool_); #preallocate
    isinBig = np.zeros((np.int64(circums*dataPtsPerCircum)+1),dtype=np.bool_); #preallocate
    
    #--- ray trace the points within all of the boxes ---
    if( (os.path.isfile('tesseraeObsPickle_'+str(k)+'.pkl') == True) & (cacheItUp == True) ): #try to load it from pre-calc'd stuff
        with open(os.getcwd()+dataFolder+'tesseraeObsPickle_'+str(k)+'.pkl','rb') as fPkl:
            obz, obz_inTot, obz_in, obz_inOrdered = pkl.load(fPkl); #load a pickle
        #END WITH
    else: #otherwise calc it
        obz_in = []; #Prep list
        obz_inTot = 0; #prep cntr
        for i in range(0,len(tbox)): #cruise through every tbox
            #--- build reference vectors ---
            #inner is a faster dot apparently
            # if( np.all(tbox[i][0,0] < tbox[i][1:,0]) == False ): #1st pt gotta be left most
            #     tbox[i] = np.roll(tbox[i],np.where(np.flipud(tbox[i])[:,0].min() == np.flipud(tbox[i])[:,0])[0][0]*2+2); #fix the orientation
            # #END IF
            V12 = np.array( ((tbox[i][0][1,0]-tbox[i][0][0,0]),(tbox[i][0][1,1]-tbox[i][0][0,1])) ); #build it
            M12 = np.sqrt(np.inner(V12,V12)); #fastest magntiude calc in the west
            V14 = np.array( ((tbox[i][0][3,0]-tbox[i][0][0,0]),(tbox[i][0][3,1]-tbox[i][0][0,1])) ); #build it
            M14 = np.sqrt(np.inner(V14,V14)); #fastest magntiude calc in the west
            V32 = np.array( ((tbox[i][0][1,0]-tbox[i][0][2,0]),(tbox[i][0][1,1]-tbox[i][0][2,1])) ); #build it
            M32 = np.sqrt(np.inner(V32,V32)); #fastest magntiude calc in the west
            V34 = np.array( ((tbox[i][0][3,0]-tbox[i][0][2,0]),(tbox[i][0][3,1]-tbox[i][0][2,1])) ); #build it
            M34 = np.sqrt(np.inner(V34,V34)); #fastest magntiude calc in the west
            #calc cos(theta) instead of theta for angles
            c1214 = np.inner(V12,V14)/(M12*M14); #cos(Theta12,14) is this, don't calc the cos for speed
            c3234 = np.inner(V32,V34)/(M32*M34); #cos(Theta32,34) is this, don't calc the cos for speed
    
            #--- make sure pts are possibly in ---
            kk = (tbox[i][0][:,0].max() >= locz[:,0]) & (tbox[i][0][:,0].min() <= locz[:,0]) & (tbox[i][0][:,1].max() >= locz[:,1]) & (tbox[i][0][:,1].min() <= locz[:,1]); #only look at pts within the current box maximal range
            if( kk.sum() > 0 ):
                V1p = np.empty((kk.sum(),2)); #preallocate
                V3p = np.empty((kk.sum(),2)); #preallocate
                #--- calc vects for every point ---
                V1p[:,0] = locz[kk,0]-tbox[i][0][0,0]; #calc long
                V1p[:,1] = locz[kk,1]-tbox[i][0][0,1]; #calc lat
                M1p = np.sqrt(np.einsum('...i,...i', V1p, V1p)); #memory error says to einsum (equiv to np.sqrt(V1p[:,0]**2+V1p[:,1]**2) but prolly faster b/c einstein
                V3p[:,0] = locz[kk,0]-tbox[i][0][2,0]; #calc long
                V3p[:,1] = locz[kk,1]-tbox[i][0][2,1]; #calc lat
                M3p = np.sqrt(np.einsum('...i,...i', V3p, V3p)); #memory error says to einsum (equiv to np.sqrt(V1p[:,0]**2+V1p[:,1]**2) but prolly faster b/c einstein
                #calc cos(theta) instead of theta for angles
                c121p = np.einsum('...i,...i', V12, V1p)/(M12*M1p);
                c141p = np.einsum('...i,...i', V14, V1p)/(M14*M1p);
                c323p = np.einsum('...i,...i', V32, V3p)/(M32*M3p);
                c343p = np.einsum('...i,...i', V34, V3p)/(M34*M3p);
                
                #--- is in? ---
                isin = ((c121p >= c1214) & (c141p >= c1214) & (c323p >= c3234) & (c343p >= c3234)); #record if it is in
                isinBig = isinBig & False; #false it out
                isinBig[kk] = isin; #put in the right spots
                # #patch b/c I don't feel like finding out what's so wrong
                # if( np.all( tbox[i][0][:,1].max() < locz[isin,1] ) | np.all( tbox[i][0][:,1].min() > locz[isin,1] ) ):
                #     isin = isin & False; #fix it with a hammer
                # #END IF
                if( isinBig.sum() > 0 ):
                    isinDiff = np.diff(isinBig); #get the diff
                    if(isinBig[0] == True):
                        isinDiff = np.insert(isinDiff,0,True); #make diff match length and catch if it starts in the area
                    else:
                        isinDiff = np.insert(isinDiff,0,False); #make diff match length
                    #END IF
                    if(isinBig[-1] == True):
                        isinDiff[-1] = True; #catch if it ends on a detection
                    #END IF
                    isinDiffWhere = np.where(isinDiff==True)[0];
                    for j in np.arange(0,isinDiff.sum(),2):
                        #catch where not enough pts to confirm tesserae
                        if( (isinDiffWhere[j+1]-isinDiffWhere[j]) < ptsInTesseraeReq ):
                            isinBig[isinDiffWhere[j]:isinDiffWhere[j+1]] = False; #set to false b/c not actually good data
                        else:
                            obz_inTot += 1; #increment
                            obz_in.append((obz_inTot,isinDiffWhere[j],isinDiffWhere[j+1],i,tbox[i][1],locz[isinDiffWhere[j]],locz[isinDiffWhere[j+1]])); #append on the number of obz in and the indexes of the data and the box it was in
                        #END IF
                    #END FOR j
                    obz = obz | isinBig; #include that
                #END IF
            #END IF
        #END FOR i
        obz_inOrder = np.empty(len(obz_in),dtype=np.int64); #prep
        for i in range(0,len(obz_in)):
            obz_inOrder[i] = obz_in[i][1]; #get the init times
        #END FOR i    
        obz_inOrderSortIdx = np.argsort(obz_inOrder); #sort them
        obz_inOrdered = []; #prep
        for i in range(0,len(obz_in)):
            obz_inOrdered.append(obz_in[obz_inOrderSortIdx[i]]); #build it
        #END FOR i
        obz_inOrder_zones = np.empty(len(obz_in),dtype=np.int64); #prep
        for i in range(0,len(obz_in)):
            obz_inOrder_zones[i] = obz_in[i][1]; #get the init times
        #END FOR i    
        obz_inOrderSortIdx = np.argsort(obz_inOrder_zones); #sort them
        obz_inOrdered_zones = []; #prep
        for i in range(0,len(obz_in)):
            obz_inOrdered_zones.append(obz_in[obz_inOrderSortIdx[i]]); #build it
        #END FOR i
        if( cacheItUp == True ):
            with open(os.getcwd()+dataFolder+'tesseraeObsPickle_'+str(k)+'.pkl', 'wb') as fPkl:
                pkl.dump([obz, obz_inTot, obz_in, obz_inOrdered], fPkl); #dump to pickle
            #END WITH
        #END IF
    #END IF
    #--- get all the needed bitz into a dict ---
    # locz_dict.append(np.copy(locz)); #copy it over
    # obz_dict['obz'].append(np.copy(obz)); #copy it over
    obz_dict['obz_inTot'].append(np.copy(obz_inTot)); #copy it over
    # obz_dict['obz_in'].append(obz_in.copy()); #copy it over
    obz_dict['obz_inOrdered_zones'].append(obz_inOrdered_zones.copy()); #copy it over
#END FOR k

#crunch - this is baaad code but it was wrote fast no judge do not judg
obz_inOrdered_zones_time = np.empty((len(platformPaths),2));
for k in range(0,len(platformPaths)):    
    cntr = 0;
    for i in range(0,len(obz_dict['obz_inOrdered_zones'][k])):
        if( cntr < obsGoal_sep ):
            if( obz_dict['obz_inOrdered_zones'][k][i][4] > 0 ):
                cntr += 1; #increment
                if( obsGoal_sep == cntr ):
                    obz_inOrdered_zones_time[k,0] = obz_dict['obz_inOrdered_zones'][k][i][5][2]/daysPerCircum+(ptsInTesseraeReq/dataRate)/(daysPerCircum*86400);
                    obz_inOrdered_zones_time[k,1] = obz_dict['obz_inOrdered_zones'][k][i][4]; #record the time
                #END IF
            #END IF
        #END IF
    #END FOR i
#END FOR k

#--- crunch some numbers! ---
def mean_confidence_interval(data, confidence=0.95): #from https://stackoverflow.com/a/15034143/2403531
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

results = mean_confidence_interval(obz_inOrdered_zones_time[:,0],confidence=0.99); #results -10 to 10 (84) (0.8221661614491476, 0.602945003751643, 1.041387319146652)
print('Circumnavigation Analysis on '+str(len(platformPaths))+' aerial paths from '+str(np.min(goodLatitudes))+' to '+str(np.max(goodLatitudes))+' latitude'+\
      '\nMean: '+str(np.round(results[0],2)).rstrip('0').rstrip('.')+' | Lower 99%: '+str(np.round(results[1],2)).rstrip('0').rstrip('.')+' | Upper 99%: '+str(np.round(results[2],2)).rstrip('0').rstrip('.')+\
      '\n99% confidence +/- '+str(np.round(results[2]-results[0],2)).rstrip('0').rstrip('.')+\
      '\nMin Circ #: '+str(np.round(np.min(obz_inOrdered_zones_time[:,0]),2)).rstrip('0').rstrip('.')+' | Max Circ #: '+str(np.round(np.max(obz_inOrdered_zones_time[:,0]),2)).rstrip('0').rstrip('.')+' | STDEV #: '+str(np.round(np.std(obz_inOrdered_zones_time[:,0]),2)).rstrip('0').rstrip('.')
      ); #report

#--- this is for seeing a specific plot out of the 84 ---
k = 7; #which to plot (7 is crazy one)
obz_dict['obz_inOrdered'] = obz_dict['obz_inOrdered_zones']; #set
platformPath_raw = platformPaths[k]; #get the path needed
platformPath_interper = interpolate.UnivariateSpline(platformPath_raw[:,0],platformPath_raw[:,1]); #use this to input any day portion and get out latitudes
#--- build a big array of everything ---
locz[:,1] = platformPath_interper(locz[:,2]); #calc the latitudes

#==============PLOT STANDARDIZATION STUFF==============
FONT_axisTick = 23; #small font (tick mark font size) Default: 19, Big: 23
FONT_axisLabel = 28; #medium font (labels) Default: 23, Big: 28
FONT_title = 28; #big font (title only really) Default: 23, Big: 28
FONT_grandiose = 32; #big font (title only really) Default: 26, Big: 32
FONT_font = 'arial';
FONT_weight = 'bold';

import matplotlib.font_manager as fm #import font manager you know
FONT_axisTickFM = fm.FontProperties(family=FONT_font, weight=FONT_weight, size=FONT_axisTick); #these are font properties, some plot stuff has this and it tells it all in one go #inconsistent
FONT_axisLabelFM = fm.FontProperties(family=FONT_font, weight=FONT_weight, size=FONT_axisLabel); #these are font properties, some plot stuff has this and it tells it all in one go #inconsistent
FONT_titleFM = fm.FontProperties(family=FONT_font, weight=FONT_weight, size=FONT_title); #these are font properties, some plot stuff has this and it tells it all in one go #inconsistent
FONT_grandioseFM = fm.FontProperties(family=FONT_font, weight=FONT_weight, size=FONT_grandiose); #these are font properties, some plot stuff has this and it tells it all in one go #inconsistent

plt.rcParams['font.weight'] = FONT_weight; #sents default font weight to bold for everything else
plt.rcParams['axes.labelweight'] = FONT_weight; #sets default font weight to bold for axis labels
plt.rc('font', size=FONT_axisTick); #default text size
plt.rc('xtick', labelsize=FONT_axisTick); #x tick label font size
plt.rc('ytick', labelsize=FONT_axisTick); #y tick label font size
plt.rc('legend', fontsize=FONT_axisLabel); #legend fornt size
plt.rc('figure', titlesize=FONT_title); #figure title font size (this one didn't do anything, so 2nd here also)
plt.rc('axes', titlesize=FONT_title); #figure title font size (this one did work)
plt.rcParams['axes.labelsize'] = FONT_axisLabel; #try this one also

#--- Visualize ---
fig, ax = plt.subplots(); #use instead of fig because it inits an axis too (I think I dunno)
figManager = plt.get_current_fig_manager(); #req to maximize
figManager.window.showMaximized(); #force maximized
#Remove the aspect ratio from the basemap so it fills the screen better
ax.set_aspect('auto');

#plot obs path
#make colorbar
divider = make_axes_locatable(ax); #prep to add an axis
cax = divider.append_axes('right', size='2.0%', pad=0.35); #make a color bar axis

im = ax.scatter(locz[0::dataRate**4,0],locz[0::dataRate**4,1],s=10,c=locz[0::dataRate**4,2]/daysPerCircum,cmap='inferno');
cbar = fig.colorbar(im, cax=cax, orientation='vertical'); #create a colorbar using the prev. defined cax
cbar.ax.tick_params(labelsize=FONT_axisTick);
# cax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f')); #force a rounded format
cbar.set_label('Circumnavigation #'); #tabel the colorbar
cbar.ax.tick_params(labelsize=FONT_axisTick);
# cbar.mappable.set_clim(vmin=np.min(plotLimValu), vmax=np.max(plotLimValu));

obz_inOrdered = obz_dict['obz_inOrdered'][k]; #pull the right one out
#plot viewed obs for verification
if( plot_toGoalOnly == False ):
    lenToGoTo = len(obz_inOrdered); #hit em all
else:
    lenToGoTo = toGoal[k]; #go to obs goal only
#END IF
for i in range(0,lenToGoTo):
    im_obs, = ax.plot(locz[obz_inOrdered[i][1]:obz_inOrdered[i][2]:dataRate**3,0],locz[obz_inOrdered[i][1]:obz_inOrdered[i][2]:dataRate**3,1],linewidth=5,c='xkcd:azure',zorder=1000);
    ax.plot(locz[obz_inOrdered[i][1]:obz_inOrdered[i][2]:dataRate**3,0],locz[obz_inOrdered[i][1]:obz_inOrdered[i][2]:dataRate**3,1],linewidth=7.5,c='xkcd:white',zorder=900);
    # ax.plot(np.concatenate( (np.array((locz[obz_inOrdered[i][1],0],)),locz[obz_inOrdered[i][1]:obz_inOrdered[i][2]:100000,0],np.array((locz[obz_inOrdered[i][2],0],))) ), \
    #         np.concatenate( (np.array((locz[obz_inOrdered[i][1],1],)),locz[obz_inOrdered[i][1]:obz_inOrdered[i][2]:100000,1],np.array((locz[obz_inOrdered[i][2],1],))) ), \
    #         marker='*',markersize=10,linewidth=3,c='xkcd:azure',zorder=1000);
#END FOR i

#plot tesserae boxes
for i in range(0,len(tbox)): #cruise through every tbox
    # i += 1; #for manual iterating through the tesserae boxes uncomment this and run this set w/o the for loop incrementally to see which is which [start at i = -1]
    temp_mapCoords = ( np.hstack( [np.linspace(tbox[i][0][0,1],tbox[i][0][1,1],20) , \
        np.linspace(tbox[i][0][1,1],tbox[i][0][2,1],20) , \
        np.linspace(tbox[i][0][2,1],tbox[i][0][3,1],20) , \
        np.linspace(tbox[i][0][3,1],tbox[i][0][0,1],20)] ) , \
        np.hstack( [np.linspace(tbox[i][0][0,0],tbox[i][0][1,0],20) , \
        np.linspace(tbox[i][0][1,0],tbox[i][0][2,0],20) , \
        np.linspace(tbox[i][0][2,0],tbox[i][0][3,0],20) , \
        np.linspace(tbox[i][0][3,0],tbox[i][0][0,0],20)] ) ); #convert to the geographic map coords
    im_tess, = ax.plot( temp_mapCoords[1],  #X longitude arcdeg
        temp_mapCoords[0],  #Y latitude arcdeg
        c='xkcd:fuchsia',linewidth=1.5, zorder=90);
#END FOR i

#plot landing point
im_landing, = ax.plot(landingCoords[0],landingCoords[1],marker='*', color='xkcd:red',linestyle='None', markersize=41,zorder=2000);
ax.plot(landingCoords[0],landingCoords[1],marker='*', color='xkcd:white', markersize=57,zorder=1999);

# #plot end of obs goal point
# for k in range(0,len(keyz)):
#     obz_inOrdered = obz_dict['obz_inOrdered']; #pull the right one out
#     locz = locz_dict; #pull the right one out
#     #plot viewed obs for verification
#     # ax.plot(locz[obz_inOrdered[obsGoal-1][2],0],locz[obz_inOrdered[obsGoal-1][2],1],marker='X', color='xkcd:fire engine red', markersize=20);
#     tx1 = ax.text(locz[obz_inOrdered[obsGoal-1][2],0],locz[obz_inOrdered[obsGoal-1][2],1],'\N{check mark}',color='xkcd:green blue',fontsize=40,zorder=1500,horizontalalignment='center',verticalalignment='center'); #put a checkmark there
#     ax.text(locz[obz_inOrdered[obsGoal-1][2],0],locz[obz_inOrdered[obsGoal-1][2],1],'\N{check mark}',color='xkcd:white',fontsize=60,zorder=1499,horizontalalignment='center',verticalalignment='center'); #put a checkmark there
# #END FOR k

leg_landing = im_landing;
leg_tess = im_tess;
leg_obs = im_obs;
leg_goal = mpatches.Patch(color='xkcd:green blue'); #tx1; 
leg = ax.legend(handles=[leg_landing,leg_tess,leg_obs],labels=['Deployment Location','Tesserae Regions','Tesserae Passover']);
leg.legendHandles[1].set_linewidth(5);

#venus image
if os.path.isfile('tesseraeMap.png'):
    img = plt.imread('tesseraeMap.png'); #read in image
    ax.imshow(img, extent=[0, 360, -90, 90]); #plot it
    ax.set_aspect('auto'); #reset aspect ratio to yolo
#END IF

#title stuff
# ax.set_title('Total # of Tesserae Observations with at least '+str(ptsInTesseraeReq)+' pts is '+str(obz_inTot)+\
#              ' for '+str(days)+' days/'+str(np.round(circums,2)).rstrip('0').rstrip('.')+' circumnavigations'+\
#              '\n'+str(np.round(locz[obz_inOrdered[obsGoal-1][2],2],2)).rstrip('0').rstrip('.')+' days/'+str(np.round(locz[obz_inOrdered[obsGoal-1][2],2]/daysPerCircum,2)).rstrip('0').rstrip('.')+' cnav to reach '+str(obsGoal)+' TessObs @ red x',fontproperties=FONT_titleFM); #set the title
# ax.set_title('Total # of Tesserae Observations with at least '+str(ptsInTesseraeReq)+' pts is '+str(obz_inTot)+\
#              ' for '+str(days)+' days/'+str(np.round(circums,2)).rstrip('0').rstrip('.')+' circumnavigations',\
#              fontproperties=FONT_titleFM,y=1.015); #set the title
#x axis stuff
ax.set_xlabel('Longitude [deg]',fontproperties=FONT_axisLabelFM);
ax.set_xlim((0,360)); #set the xlims now
ax.set_xticks(np.arange(0,360+30,30)); #set xticks
labelz = ax.get_xticklabels(); #get x tick labels
labelzNew = np.arange(-180,180+30,30);
for i in range(0,len(labelz)):
    labelz[i].set_text(str(labelzNew[i])); #convert from 0to360 to -180to180
#END FOR i
ax.set_xticklabels(labelz); #set the labelz
#y axis stuff
ax.set_ylabel('Latitude [deg]',fontproperties=FONT_axisLabelFM);
ax.set_ylim((-plot_latLim,plot_latLim)); #set the ylims now
ax.set_yticks(np.arange(-plot_latLim,plot_latLim+plot_latSpacing,plot_latSpacing)); #set yticks
#fit that fig fast
try:
    figFitter(fig); #fit that fig
except:
    fig.tight_layout(); #fit that fig less fit
#END TRY

obz_inTot = obz_dict['obz_inTot'][k]; #pull the right one out
obz_inOrdered = obz_dict['obz_inOrdered'][k]; #pull the right one out
print('Total # of Tesserae Observations with at least '+str(ptsInTesseraeReq)+' pts for '+keyz[k]+' path is '+str(obz_inTot)+\
    ' for '+str(days)+' days/'+str(np.round(circums,2)).rstrip('0').rstrip('.')+' circumnavigations'+\
    '\n'+str(np.round(locz[obz_inOrdered[obsGoal-1][2],2],2)).rstrip('0').rstrip('.')+' days/'+str(np.round(locz[obz_inOrdered[obsGoal-1][2],2]/daysPerCircum,2)).rstrip('0').rstrip('.')+' circumnavigations to reach '+str(obsGoal)+' Tesserae Observations\n')


