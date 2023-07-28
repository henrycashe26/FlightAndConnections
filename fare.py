#money
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df_fare = pd.read_csv('itineraries3.csv', low_memory=False)
# print(df_fare.head())
# print(df_fare.shape[0])


# #predicting if it's non stop
# X = df_fare[['baseFare', 'totalFare', 'totalTravelDistance']]
# Y = df_fare["isNonStop"]
# x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
# model = KNeighborsClassifier()
# model.fit(x_train, y_train)
# predictions = model.predict(x_test)
# accuracy = accuracy_score(predictions, y_test)
# print(accuracy)



# #assingn airlines values, and airports values
# #hash each airport is stored to a unique value first come first serve
Airport_codes = {
    "BO": [937,'42°21′51″N, 71°0′18″W'],
    'ATL':[0, '33°38′12″N, 84°25′41″W'],
    'BOS':[937,'42°21′51″N, 71°0′18″W'],
    'CLT': [259, "35°12′50″N, 80°56′35″W"],
    'DEN': [1199, "39°51′42″N, 104°40′22″W"],
    'DFW': [731, "32°53′48″N, 97°2′16″W"],
    'DTW': [594, "42°12′44″N, 83°21′12″W"],
    'EWR': [762, "40°41′33″N, 74°10′7″W"],
    'IAD': [547, "38°56′40″N, 77°27′20″W"],
    'JFK': [762, "40°38′23″N, 73°46′44″W"],
    'LAX': [1947, "33°56′33″N, 118°24′28″W"],
    'LGA': [762, "40°46′37″N, 73°52′21″W"],
    'MIA': [594, "25°47′35″N, 80°17′26″W"],
    "OAK": [2130, "33°38′12″N, 84°25′41″W"],
    'ORD': [606, "41°58′42″N, 87°54′17″W"],
    "PHL": [666, "39°52′18″N, 75°14′27″W"],
    "SFO": [2132, "39°52′18″N, 75°14′27″W"],
    "SYR": [794, "43°6′40″N, 76°6′22″W"],
    "DE": [1199, "39°51′42″N, 104°40′22″W"],
    "IAH":[689, "29°59′3″N, 95°20′29″W"],
    "MCO":[404, "28°25′45″N, 81°18′32″W"],
    "DA":[366, "39°54′8″N, 84°13′9″W"],
    "DF":[1330, "19°26′10″N, 99°4′19″W"],
    "TPA": [406, "27°58′31″N, 82°31′59″W"],
    "DT": [594, "42°12′44″N, 83°21′12″W"],
    "CLE": [553, "41°24′42″N, 81°50′59″W"],
    "MYR": [317, "33°40′46″N, 78°55′41″W"],
    "PBI": [545, "26°40′59″N, 80°5′44″W"],
    "PIT": [526, "40°29′29″N, 80°13′58″W"],
    "PWM":[1027, "43°38′46″N, 70°18′33″W"],
    "SRQ": [444, "27°23′43″N, 82°33′15″W"],
    "IND":[432, "39°43′2″N, 86°17′39″W"],
    "JAX":[270, "30°29′38″N, 81°41′16″W"],
    "MCI":[692, "39°17′51″N, 94°42′50″W"],
    "CHA":[106, "35°2′7″N, 85°12′13″W"],
    "BNA":[214, "36°7′28″N, 86°40′41″W"],
    "OWB": [322, "37°44′24″N, 87°10′0″W"],
    "LEX": [304, "38°2′11″N, 84°36′21″W"],
    "GSO": [306, "36°5′52″N, 79°56′14″W"],
    "ROC": [749, "43°7′8″N, 77°40′20″W"],
    "EW":[746, "40°41′33″N, 74°10′7″W"],
    "IA": [694, "41°53′4″N, 91°42′38″W"],
    "CMH": [447, "39°59′52″N, 82°53′30″W"],
    "STL": [484, "38°44′55″N, 90°22′12″W"],
    "JF": [762, "40°38′23″N, 73°46′44″W"],
    "LAS":[1747, "33°38′12″N, 84°25′41″W"],
    "DEN": [1747, "39°51′42″N, 104°40′22″W"],
    "PHX": [1587, "33°26′3″N, 112°0′43″W"],
    "ON": [1899, "34°3′21″N, 117°36′3″W"],
    "SLC": [1590, "40°47′18″N, 111°58′40″W"],
    "AUS": [813, "30°11′40″N, 97°40′11″W"],
    "SAT": [874, "29°32′1″N, 98°28′11″W"],
    "LA": [1946, "33°56′33″N, 118°24′28″W"],
    "TUS": [1541, "33°38′12″N, 84°25′41″W"],
    "MSP": [907, "44°52′55″N, 93°13′18″W"],
    "LG": [762,"40°46′37″N, 73°52′21″W"],
    "MI":[594, "42°12′44″N, 83°21′12″W"],
    "LAX":[1946, "33°56′33″N, 118°24′28″W"],
    "SEA":[2182, "47°26′56″N, 122°18′32″W"],
    "OR":[516, "36°53′40″N, 76°12′4″W"],
    "PH":[666,"39°52′18″N, 75°14′27″W"],
    "SF":[2132, "39°52′18″N, 75°14′27″W"],
    "FIL":[666,"39°52′18″N, 75°14′27″W"],
    "AT":[0, '33°38′12″N, 84°25′41″W'],
    "PHL":[666,"39°52′18″N, 75°14′27″W"],
    "MSS":[932, "44°56′8″N, 74°50′44″W"],
    "CVG":[373, "39°2′55″N, 84°40′4″W"],
    "RDU":[356, "35°52′39″N, 78°47′14″W"],
    "ACK":[947,"41°15′11″N, 70°3′36″W"],
    "BGR":[1134, "44°48′26″N, 68°49′41″W"],
    "SLK":[922, "44°23′7″N, 74°12′22″W"],
    "DCA":[547, "38°51′7″N, 77°2′15″W"],
    "SAN":[1892, "32°44′0″N, 117°11′24″W"],
    "MOB":[302,"30°41′28″N, 88°14′34″W"],
    "HSV":[151, "34°38′13″N, 86°46′30″W"],
    "GSP":[153, "34°53′44″N, 82°13′8″W"],
    "LIT":[453, "34°43′45″N, 92°13′27″W"],
    "CHO":[457, "38°8′18″N, 78°27′10″W"],
    "ROA":[357, "37°19′31″N, 79°58′31″W"],
    "DAY":[432,"39°54′8″N, 84°13′9″W"],
    "MKE":[669, "42°56′49″N, 87°53′47″W"],
    "FWA":[508, "40°58′42″N, 85°11′42″W"],
    "MDT":[620, "40°11′36″N, 76°45′48″W"],
    "SNA":[1919, "33°40′32″N, 117°52′4″W"],
    "CHS":[259, "32°53′54″N, 80°2′25″W"],
    "BWI": [577, "39°10′31″N, 76°40′5″W"],
    "RIC":[481, "37°30′18″N, 77°19′10″W"],
    "ORF":[516, "36°53′40″N, 76°12′4″W"],
    "TEX":[696, "29°38′43″N, 95°16′44″W"],
    "OA":[2130, "33°38′12″N, 84°25′41″W"],
    "PDX":[2172, "45°35′19″N, 122°35′52″W"], 
    "GTF":[1696, "47°28′55″N, 111°22′15″W"],
    "DSM": [743, "41°32′2″N, 93°39′47″W"],
    "DAL":[731, "32°53′48″N, 97°2′16″W"],
    "TYS":[152, "35°48′39″N, 83°59′38″W"],
    "AVL":[164, "33°38′12″N, 84°25′41″W"],
    "MGM":[147, "32°18′2″N, 86°23′38″W"],
    "GLH":[378, "33°28′58″N, 90°59′8″W"],
    "CVN":[373, "39°2′55″N, 84°40′4″W"],
    "BZN":[1640, "45°46′39″N, 111°9′10″W"],
    "XNA":[589, "36°16′54″N, 94°18′24″W"],
    "OKC":[761, "35°23′35″N, 97°36′2″W"],
    "SJC":[2116, "37°21′45″N, 121°55′44″W"],
    "PNS":[271, "30°28′24″N, 87°11′11″W"],
    "MSY":[425, "29°59′36″N, 90°15′28″W"],
    "SGF":[563, "37°14′44″N, 93°23′18″W"],
    "TUL":[674, "36°11′54″N, 95°53′17″W"],
    "BHM":[134, "33°33′46″N, 86°45′12″W"],
    "MSN":[707, "43°8′23″N, 89°20′15″W"],
    "MEM":[332, "35°2′32″N, 89°58′36″W"],
    "ABE":[692, "40°39′7″N, 75°26′26″W"],
    "ALB":[853, "42°44′53″N, 73°48′6″W"],
    "MVY":[927, "33°38′12″N, 84°25′41″W"],
    "LNS":[634, "33°38′12″N, 84°25′41″W"],
    "DUJ":[602, "41°10′41″N, 78°53′55″W"],
    "ONT":[1899, "34°3′21″N, 117°36′3″W"],
    "ABQ":[1269, "35°2′24″N, 106°36′32″W"],
    "BOI":[1838, "33°38′12″N, 84°25′41″W"],
    "SDF":[321, "38°10′27″N, 85°44′9″W"],
    "TLH":[223, "30°23′47″N, 84°21′1″W"],
    "SAV":[214, "32°7′39″N, 81°12′7″W"],
    "OAK": [2130, "33°38′12″N, 84°25′41″W"],
    "COS":[1184, "33°38′12″N, 84°25′41″W"],
    "ATY":[1032, "44°54′50″N, 97°9′16″W"],
    "MHK":[780, "39°8′27″N, 96°40′14″W"],
    "BMI":[533, "33°38′12″N, 84°25′41″W"],
    "CID":[694, "41°53′4″N, 91°42′38″W"],
    "BTV":[961, "44°28′18″N, 73°9′11″W"],
    "OMA":[821, "41°18′11″N, 95°53′38″W"],
    "GEG":[1969, "47°37′11″N, 117°32′2″W"],
    "TTN":[701, "40°16′36″N, 74°48′48″W"],
    "ILM":[377, "34°16′14″N, 77°54′9″W"],
    "GRR":[640, "42°52′50″N, 85°31′22″W"],
    "PSC":[2016, "46°15′52″N, 119°7′8″W"],
    "PSP":[1840, "33°49′46″N, 116°30′25″W"],
    "PHF":[508, "37°7′54″N, 76°29′34″W"],
    "SHV":[551, "32°26′47″N, 93°49′32″W"],
    "CEZ":[694, "41°53′4″N, 91°42′38″W"],
    "COU":[1184, "33°38′12″N, 84°25′41″W"],
    "BDL":[859, "41°56′20″N, 72°40′59″W"],
    "RNO":[1993, "33°38′12″N, 84°25′41″W"],
    "MWA":[340, "32°18′40″N, 90°4′33″W"],
    "EVV":[350, "33°38′12″N, 84°25′41″W"],
    "ELP":[1282, "31°48′25″N, 106°22′40″W"],
    "UIN":[574, "33°38′12″N, 84°25′41″W"],
    "RSW":[515, "26°32′10″N, 81°45′18″W"],
    "CAE":[192, "33°38′12″N, 84°25′41″W"],
    "CNM":[1154, "32°20′15″N, 104°15′46″W"],
    "MGW":[484, "39°38′34″N, 79°54′58″W"],
    "TT":[701, "40°16′36″N, 74°48′48″W"],
    "PVD":[904,"41°43′57″N, 71°25′13″W"],
    "BFD":[646, "33°38′12″N, 84°25′41″W"],
    "MFR":[2163, "42°22′27″N, 122°52′22″W"],
    "MKL":[289, "33°38′12″N, 84°25′41″W"],
    "FLL":[581, "26°4′21″N, 80°9′9″W"],
    "EUG":[2189, "44°7′28″N, 123°12′43″W"]
}


def unpackCoordinates(coord):
    answer = []
    answer.append(coord[:coord.find("°")])
    coord = coord[coord.find("°")+1:]
    answer.append(coord[:coord.find("′")])
    coord = coord[coord.find("′")+1:]
    answer.append(coord[:coord.find("″")])
    coord = coord[coord.find("″")+1:]
    answer.append(coord[coord.find(" "):coord.find("°")])
    coord = coord[coord.find("°")+1:]
    answer.append(coord[:coord.find("′")])
    coord = coord[coord.find("′")+1:]
    answer.append(coord[:coord.find("″")])
    coord = coord[coord.find("″")+1:]
    return answer

print("ANSWER")
print(unpackCoordinates("40°16′36″N, 74°48′48″W"))

Airline_codes = {}
running_Value = 1
startingAirportDistanceFromAtlanta = []
destinationAirportDistanceFromAtlanta = []
minutesDuration = []
segmentsDistanceList = [[],[],[],[]]
segmentsDurationList = [[],[],[],[]]
segmentsEpochList = [[],[],[],[]]
cordinates = [[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
segmentCoordinates = [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
segmentDistance=[[],[],[]]
segmentAirportDistance=[[],[],[]]
for index, row in df_fare.iterrows():
    #segmentAirports = []
    airport = row['startingAirport']
    destination = row['destinationAirport']
    duration = row["travelDuration"]
    segmentsDistance = row["segmentsDistance"]
    segmentsDuration = row["segmentsDurationInSeconds"]
    segmentsEpoch = row["segmentsDepartureTimeEpochSeconds"]
    # #if(row["segmentsAirlineName"].contains("||"))
    # if(Airport_codes.get(airport) == None):
    #     Airport_codes[airport] = running_Value
    #     running_Value+=1
    startingAirportDistanceFromAtlanta.append(Airport_codes[airport][0])
    # if(Airport_codes.get(destination) == None):
    #     Airport_codes[destination] = running_Value
    #     running_Value+=1
    destinationAirportDistanceFromAtlanta.append(Airport_codes[destination][0])
    minutes = 0
    if(duration.find("M")!=-1):
        if(duration.find("H")!=-1):
            if(duration[duration.find("H")+1:duration.find("M")] != ''):
                minutes += int(duration[duration.find("H")+1:duration.find("M")])
        else:
            if(duration[duration.find("T")+1:duration.find("M")] != ''):
                minutes += int(duration[duration.find("T")+1:duration.find("M")])
    if(duration[duration.find("T")+1:duration.find("H")] != ''):
        minutes += int(duration[duration.find("T")+1:duration.find("H")])*60
    minutesDuration.append(minutes)
    
    segmentsDistanceArray = [0,0,0,0]
    indexDistance = 0
    while(segmentsDistance.__contains__("||")):
        segmentsDistanceArray[indexDistance] = (int(segmentsDistance[:segmentsDistance.find("||")]))
        segmentsDistance = segmentsDistance[segmentsDistance.find("||")+2:]
        indexDistance+=1
    segmentsDistanceArray[indexDistance] = int(segmentsDistance)
    segmentsDistanceList[0].append(segmentsDistanceArray[0])
    segmentsDistanceList[1].append(segmentsDistanceArray[1])
    segmentsDistanceList[2].append(segmentsDistanceArray[2])
    segmentsDistanceList[3].append(segmentsDistanceArray[3])

    segmentsDurationArray = [0,0,0,0]
    indexDuration = 0
    while(segmentsDuration.__contains__("||")):
        segmentsDurationArray[indexDuration] = (int(segmentsDuration[:segmentsDuration.find("||")]))
        segmentsDuration = segmentsDuration[segmentsDuration.find("||")+2:]
        indexDuration+=1
    segmentsDurationArray[indexDuration] = int(segmentsDuration)
    segmentsDurationList[0].append(segmentsDurationArray[0])
    segmentsDurationList[1].append(segmentsDurationArray[1])
    segmentsDurationList[2].append(segmentsDurationArray[2])
    segmentsDurationList[3].append(segmentsDurationArray[3])
    
    segmentsEpochArray = [0,0,0,0]
    indexEpoch = 0
    while(segmentsEpoch.__contains__("||")):
        segmentsEpochArray[indexEpoch] = (int(segmentsEpoch[:segmentsEpoch.find("||")]))
        segmentsEpoch = segmentsEpoch[segmentsEpoch.find("||")+2:]
        indexEpoch+=1
    segmentsEpochArray[indexEpoch] = int(segmentsEpoch)
    segmentsEpochList[0].append(segmentsEpochArray[0])
    segmentsEpochList[1].append(segmentsEpochArray[1])
    segmentsEpochList[2].append(segmentsEpochArray[2])
    segmentsEpochList[3].append(segmentsEpochArray[3])

    airporty = row['startingAirport']
    cordinates[0][0].append(unpackCoordinates(Airport_codes[airporty][1])[0])
    cordinates[0][1].append(unpackCoordinates(Airport_codes[airporty][1])[1])
    cordinates[0][2].append(unpackCoordinates(Airport_codes[airporty][1])[2])
    cordinates[0][3].append(unpackCoordinates(Airport_codes[airporty][1])[3])
    cordinates[0][4].append(unpackCoordinates(Airport_codes[airporty][1])[4])
    cordinates[0][5].append(unpackCoordinates(Airport_codes[airporty][1])[5])
    destinationy = row['destinationAirport']
    cordinates[1][0].append(unpackCoordinates(Airport_codes[destinationy][1])[0])
    cordinates[1][1].append(unpackCoordinates(Airport_codes[destinationy][1])[1])
    cordinates[1][2].append(unpackCoordinates(Airport_codes[destinationy][1])[2])
    cordinates[1][3].append(unpackCoordinates(Airport_codes[destinationy][1])[3])
    cordinates[1][4].append(unpackCoordinates(Airport_codes[destinationy][1])[4])
    cordinates[1][5].append(unpackCoordinates(Airport_codes[destinationy][1])[5])


    segmentArriving = row["segmentsArrivalAirportCode"]
    it = 0
    if(segmentArriving.find("||") != -1):
        segmentArriving = segmentArriving[segmentArriving.find("||"):]
        while(segmentArriving.__contains__("||")):
            segmentAirport = segmentArriving[segmentArriving.find("||")+2:]
            if(segmentAirport.find("||")!=-1):
                segmentAirport = segmentAirport[:segmentAirport.find("||")]
            segmentDistance[it].append((Airport_codes[segmentAirport])[0])
            segmentAirportDistance[it].append(segmentAirport)
            segmentCoordinates[it][0].append(unpackCoordinates((Airport_codes[segmentAirport])[1])[0])
            segmentCoordinates[it][1].append(unpackCoordinates((Airport_codes[segmentAirport])[1])[1])
            segmentCoordinates[it][2].append(unpackCoordinates((Airport_codes[segmentAirport])[1])[2])
            segmentCoordinates[it][3].append(unpackCoordinates((Airport_codes[segmentAirport])[1])[3])
            segmentCoordinates[it][4].append(unpackCoordinates((Airport_codes[segmentAirport])[1])[4])
            segmentCoordinates[it][5].append(unpackCoordinates((Airport_codes[segmentAirport])[1])[5])
            it+=1
            segmentArriving = segmentArriving[segmentArriving.find("||")+2:]
    while(it<=2):
        segmentDistance[it].append(0)
        segmentAirportDistance[it].append("0")
        segmentCoordinates[it][0].append(0)
        segmentCoordinates[it][1].append(0)
        segmentCoordinates[it][2].append(0)
        segmentCoordinates[it][3].append(0)
        segmentCoordinates[it][4].append(0)
        segmentCoordinates[it][5].append(0)
        it+=1



    # largestValueDistance = 0
    # while(segmentsDistance.__contains__("||")):
    #     if(int(segmentsDistance[:segmentsDistance.find("||")]) > largestValueDistance):
    #         largestValueDistance = int(segmentsDistance[:segmentsDistance.find("||")])
    #     segmentsDistance = segmentsDistance[segmentsDistance.find("||")+2:]
    # segmentsDistanceList.append(largestValueDistance)

    # largestValueDuration = 0
    # while(segmentsDuration.__contains__("||")):
    #     if(int(segmentsDuration[:segmentsDuration.find("||")]) > largestValueDuration):
    #         largestValueDuration = int(segmentsDuration[:segmentsDuration.find("||")])
    #     segmentsDuration = segmentsDuration[segmentsDuration.find("||")+2:]
    # segmentsDurationList.append(largestValueDuration)


df_fare["startingAirportDistance"] = startingAirportDistanceFromAtlanta
df_fare["destinationAirportDistance"] = destinationAirportDistanceFromAtlanta
df_fare["minutes"] = minutesDuration
# df_fare["segmentsDistanceLargest"] = segmentsDistanceList
# df_fare["segmentsDurationInSecondsLargest"] = segmentsDurationList
df_fare["segmentsDistance1"] = segmentsDistanceList[0]
df_fare["segmentsDistance2"] = segmentsDistanceList[1]
df_fare["segmentsDistance3"] = segmentsDistanceList[2]
df_fare["segmentsDistance4"] = segmentsDistanceList[3]
df_fare["segmentsDurationInSeconds1"] = segmentsDurationList[0]
df_fare["segmentsDurationInSeconds2"] = segmentsDurationList[1]
df_fare["segmentsDurationInSeconds3"] = segmentsDurationList[2]
df_fare["segmentsDurationInSeconds4"] = segmentsDurationList[3]
df_fare["segmentsEpochTimeInSeconds1"] = segmentsEpochList[0]
df_fare["segmentsEpochTimeInSeconds2"] = segmentsEpochList[1]
df_fare["segmentsEpochTimeInSeconds3"] = segmentsEpochList[2]
df_fare["segmentsEpochTimeInSeconds4"] = segmentsEpochList[3]

df_fare["startingCoordinates1"] = cordinates[0][0]
df_fare["startingCoordinates2"] = cordinates[0][1]
df_fare["startingCoordinates3"] = cordinates[0][2]
df_fare["startingCoordinates4"] = cordinates[0][3]
df_fare["startingCoordinates5"] = cordinates[0][4]
df_fare["startingCoordinates6"] = cordinates[0][5]

df_fare["segment1Airport"] = segmentAirportDistance[0]
df_fare["segment1Coordinates1"] = segmentCoordinates[0][0]
df_fare["segment1Coordinates2"] = segmentCoordinates[0][1]
df_fare["segment1Coordinates3"] = segmentCoordinates[0][2]
df_fare["segment1Coordinates4"] = segmentCoordinates[0][3]
df_fare["segment1Coordinates5"] = segmentCoordinates[0][4]
df_fare["segment1Coordinates6"] = segmentCoordinates[0][5]
df_fare["segment1Distance"] = segmentDistance[0]

df_fare["segment2Airport"] = segmentAirportDistance[1]
df_fare["segment2Coordinates1"] = segmentCoordinates[1][0]
df_fare["segment2Coordinates2"] = segmentCoordinates[1][1]
df_fare["segment2Coordinates3"] = segmentCoordinates[1][2]
df_fare["segment2Coordinates4"] = segmentCoordinates[1][3]
df_fare["segment2Coordinates5"] = segmentCoordinates[1][4]
df_fare["segment2Coordinates6"] = segmentCoordinates[1][5]
df_fare["segment1Distance"] = segmentDistance[1]

df_fare["segment3Airport"] = segmentAirportDistance[2]
df_fare["segment3Coordinates1"] = segmentCoordinates[2][0]
df_fare["segment3Coordinates2"] = segmentCoordinates[2][1]
df_fare["segment3Coordinates3"] = segmentCoordinates[2][2]
df_fare["segment3Coordinates4"] = segmentCoordinates[2][3]
df_fare["segment3Coordinates5"] = segmentCoordinates[2][4]
df_fare["segment3Coordinates6"] = segmentCoordinates[2][5]
df_fare["segment1Distance"] = segmentDistance[2]

df_fare["destinationCoordinates1"] = cordinates[1][0]
df_fare["destinationCoordinates2"] = cordinates[1][1]
df_fare["destinationCoordinates3"] = cordinates[1][2]
df_fare["destinationCoordinates4"] = cordinates[1][3]
df_fare["destinationCoordinates5"] = cordinates[1][4]
df_fare["destinationCoordinates6"] = cordinates[1][5]

# df_fare.to_csv("UPDATEDitineries-airportCodes2.csv", index=True)
# print(Airport_codes)

#df_fare["connections"] = 0
AllAirports = {}
for index, row in df_fare.iterrows():
    segmentArriving = row["segmentsArrivalAirportCode"]
    segmentDeparting = row["segmentsDepartureAirportCode"]
    while(segmentDeparting.__contains__("||")):
        departing = segmentDeparting[:segmentDeparting.find("||")]
        if(AllAirports.get(departing) == None):
            AllAirports[departing] = 1
        segmentDeparting = segmentDeparting[segmentDeparting.find("||")+2:]
    if(AllAirports.get(segmentArriving[:segmentArriving.find("||")]) == None):
        AllAirports[segmentArriving[:segmentArriving.find("||")]] = 1



connectionsList = []
for index, row in df_fare.iterrows():
    segments = row['segmentsArrivalAirportCode']
    connections = 0
    while(segments.__contains__("||")):
        connections+=1
        segments = segments[segments.find("||")+2:]
    connectionsList.append(connections)
df_fare["connections"] = connectionsList


print(df_fare)
print(AllAirports)
df_fare.to_csv("UPDATEDitineries-airportCodes3.csv", index=True)
