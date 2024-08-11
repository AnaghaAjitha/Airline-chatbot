from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain.globals import set_verbose
set_verbose(False)
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


sys_template = """You are a helpful Airline Chatbot assistant who helps customers with any queries they have about their flight. 
You have access to the chat history {chat_history}.
You have access to the following details {context}.
If the user has not provided all three details (source, destination, and travel class), prompt the user to provide the missing details.
If all three details are provided, proceed to answer the user's question.

You can recognize the flight as international or domestic based on the source and destination.
Once you have all the information, then only proceed to answer user questions.
"""


context="""Have you checked your baggage?
If you’re someone who values bringing all your essentials on board, discover all you need to know about checked baggage allowances for a seamless air travel experience on our domestic and international flights.
Domestic Flights: {
    Economy = {
    "Comfort": "15 kg/33 lb",
    "Comfort Plus": "15 kg/33 lb",
    "Flex": "25 kg/55.1 lb"
    }
    Premium_Economy = {
        "Comfort Plus": "15 kg/33 lb",
        "Flex": "25 kg/55.1 lb"
    }
    Business = {
        "Comfort Plus": "25 kg/55.1 lb",
        "Flex": "35 kg/77.1 lb"
    }
    First = {
        "First": "40 kg/88.1 lb"
    }
}
Points to remember:

-The above allowances apply only to Air India-operated flights and not to Air India Express or codeshare flights. 
-Passengers flying on Air India domestic or international sectors on the same ticket will have the baggage allowance of the international sector applied.
-Passengers holding separate tickets for Air India domestic and connecting international flights can take advantage of Free Baggage Allowance (FBA), if:
    The connection flight is within 24 hours.
    Both flights are operated by Air India.
-Star Alliance Gold members can carry an additional 20 kg/44 lb of baggage in economy class.
-Members of premium clubs and our Flying Returns programme will get additional baggage.
-Infants are entitled to one collapsible stroller/carrycot/infant car seat.
-The maximum weight permissible for a single piece of baggage is 32 kg/71 lb. This rule applies to the entire Air India network.
-Passengers can carry assistive devices free of charge as additional baggage subject to the limitation of the aircraft as per Directorate General of Civil Aviation (DGCA) guidelines.

InternationalFlights = {
    "India (IN)": {
        "SriLanka (LK)": {
            "Economy": {
                "Comfort": "30 kg/66.1 lb",
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "40 kg/88.1 lb"
            },
            "PremiumEconomy": {
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "40 kg/88.1 lb"
            },
            "Business": {
                "ComfortPlus": "40 kg/88.1 lb",
                "Flex": "45 kg/99.2 lb"
            },
            "First": {
                "First": "45 kg/99.2 lb"
            }
        },
        "Bangladesh (BD)": {
            "Economy": {
                "Comfort": "30 kg/66.1 lb",
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "35 kg/77.1 lb"
            },
            "PremiumEconomy": {
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "35 kg/77.1 lb"
            },
            "Business": {
                "ComfortPlus": "35 kg/77.1 lb",
                "Flex": "40 kg/88.1 lb"
            },
            "First": {
                "First": "40 kg/88.1 lb"
            }
        },
        "Nepal, Myanmar, Maldives": {
            "Economy": {
                "Comfort": "20 kg/44 lb",
                "ComfortPlus": "20 kg/44 lb",
                "Flex": "30 kg/66.1 lb"
            },
            "PremiumEconomy": {
                "ComfortPlus": "20 kg/44 lb",
                "Flex": "30 kg/66.1 lb"
            },
            "Business": {
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "35 kg/77.1 lb"
            },
            "First": {
                "First": "40 kg/88.1 lb"
            }
        },
        "Israel": {
            "Economy": {
                "Comfort": "Two pieces of up to 23 kg/50.7 lb each",
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 23 kg/50.7 lb each"
            },
            "PremiumEconomy": {
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "Business": {
                "ComfortPlus": "Two pieces of up to 32 kg/70.5 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "First": {
                "First": "Two pieces of up to 32 kg/70.5 lb each"
            }
        },
        "Middle East Gulf Region": {
            "Economy": {
                "Comfort": "25 kg/55.1 lb",
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "35 kg/77.1 lb"
            },
            "PremiumEconomy": {
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "40 kg/88.1 lb"
            },
            "Business": {
                "ComfortPlus": "40 kg/88.1 lb",
                "Flex": "45 kg/99.2 lb"
            },
            "First": {
                "First": "45 kg/99.2 lb"
            }
        },
        "Thailand (TH)/ Singapore (SG)/ HongKong (HK)": {
            "Economy": {
                "Comfort": "25 kg/55.1 lb",
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "35 kg/77.1 lb"
            },
            "PremiumEconomy": {
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "40 kg/88.1 lb"
            },
            "Business": {
                "ComfortPlus": "40 kg/88.1 lb",
                "Flex": "45 kg/99.2 lb"
            },
            "First": {
                "First": "45 kg/99.2 lb"
            }
        },
        "Japan (JP)/ SouthKorea (KR)": {
            "Economy": {
                "Comfort": "Two pieces of up to 23 kg/50.7 lb each",
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 23 kg/50.7 lb each"
            },
            "PremiumEconomy": {
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "Business": {
                "ComfortPlus": "Two pieces of up to 32 kg/70.5 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "First": {
                "First": "Two pieces of up to 32 kg/70.5 lb each"
            }
        },
        "Australia and NewZealand": {
            "Economy": {
                "Comfort": "25 kg/55.1 lb",
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "35 kg/77.1 lb"
            },
            "PremiumEconomy": {
                "ComfortPlus": "30 kg/66.1 lb",
                "Flex": "40 kg/88.1 lb"
            },
            "Business": {
                "ComfortPlus": "40 kg/88.1 lb",
                "Flex": "45 kg/99.2 lb"
            },
            "First": {
                "First": "45 kg/99.2 lb"
            }
        },
        "Europe": {
            "Economy": {
                "Comfort": "Two pieces of up to 23 kg/50.7 lb each",
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 23 kg/50.7 lb each"
            },
            "PremiumEconomy": {
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "Business": {
                "ComfortPlus": "Two pieces of up to 32 kg/70.5 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "First": {
                "First": "Two pieces of up to 32 kg/70.5 lb each"
            }
        },
        "Africa": {
            "Economy": {
                "Comfort": "Two pieces of up to 23 kg/50.7 lb each",
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 23 kg/50.7 lb each"
            },
            "PremiumEconomy": {
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "Business": {
                "ComfortPlus": "Two pieces of up to 32 kg/70.5 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "First": {
                "First": "Two pieces of up to 32 kg/70.5 lb each"
            }
        },
        "UnitedStates and Canada": {
            "Economy": {
                "Comfort": "Two pieces of up to 23 kg/50.7 lb each",
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 23 kg/50.7 lb each"
            },
            "PremiumEconomy": {
                "ComfortPlus": "Two pieces of up to 23 kg/50.7 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "Business": {
                "ComfortPlus": "Two pieces of up to 32 kg/70.5 lb each",
                "Flex": "Two pieces of up to 32 kg/70.5 lb each"
            },
            "First": {
                "First": "Two pieces of up to 32 kg/70.5 lb each"
            }
        }
    }
}
}


PieceAndWeightConcept = [
    {
        "CabinClass": "First/Business",
        "MaximumSumOfDimensions": "157 cm/62 in",
        "MaximumWeightAllowedPerPiece": "32 kg/71 lb"
    },
    {
        "CabinClass": "Economy",
        "MaximumSumOfDimensions": "271 cm/107 in (both pieces combined)",
        "MaximumWeightAllowedPerPiece": "23 kg/51 lb"
    }
]


AdditionalAllowanceForFlyingReturnsMembers = {
    "description": "Flying Returns members can bring additional baggage on flights operated by Air India and Star Alliance member airlines. The additional allowance is based on the tier status of Flying Returns members for weight concept and piece concept sectors.",
    "tiers": [
        {
            "tier": "Platinum (Star Gold)",
            "piece_concept_flights": "One piece up to 23 kg",
            "weight_concept_flights": "20 kg/44 lb"
        },
        {
            "tier": "Gold (Star Gold)",
            "piece_concept_flights": "One piece up to 23 kg",
            "weight_concept_flights": "20 kg/44 lb"
        },
        {
            "tier": "Silver",
            "piece_concept_flights": "Nil",
            "weight_concept_flights": "10 kg/22 lb"
        },
        {
            "tier": "Red",
            "piece_concept_flights": "Nil",
            "weight_concept_flights": "Nil"
        }
    ]
}
Terms and conditions apply. Extra baggage allowance and lounge access is subject to change as per government directive.

Baggage rules for an interline journey
As per the International Air Transport Association (IATA), baggage rules for an interline journey are as follows:

Step 1: If the published provisions for baggage allowance among all participating carriers are the same, then those provisions will apply.

Step 2: When one or more published baggage provisions differ between participating carriers, apply any common provisions. When all provisions differ, apply the published baggage provisions of the Most Significant Carrier (MSC). In the case of codeshare flights, this will be the operating carrier unless that carrier publishes a rule stipulating that it will be the marketing carrier.

Step 3: If the MSC does not publish baggage provisions for the journey concerned, apply the published baggage provisions of the carrier accepting the baggage at check-in.

Step 4: Determine the MSC:
-For travel between two or more IATA areas, the carrier on the first sector crosses from one area to another. 
-Exception: IATA Area 1/2/3 only, the carrier providing carriage on the first sector crosses between IATA Area 1 and IATA Area 2.
-For travel between IATA tariff sub-areas, the carrier in the first sector crosses from one sub-area to another.
-For travel within an IATA tariff sub-area, it is the carrier in the first international sector.
When a passenger chooses to take a stopover at a connecting point, a new baggage assessment and charges may apply from the point of departure following the stopover.
Here are some more things to keep in mind:
-Infants in all classes are eligible for baggage allowance of up to 10 kg/22 lb of one piece.
-The maximum weight permissible for a single piece of baggage is 32 kg/71 lb. This rule applies to the entire Air India network.
-For tickets issued effective 01 April 2019, free baggage allowance in economy class between India and Newark (EWR) is one piece of up to 23 kg/51 lb.


Baggage allowance codeshare flights: {
"note": "Please refer to their websites for free baggage allowances, excess baggage charges, and optional/ancillary fees charged by our codeshare partners operating from and to the USA.",
"flights": [
    {
    "airline": "Lufthansa",
    "routes": "Chicago/Denver/Detroit/Los Angeles/Washington, DC-Frankfurt and vice versa, Frankfurt-India and vice versa.",
    "link": "Click here"
    },
    {
    "airline": "Singapore Airlines",
    "routes": "Los Angeles/San Francisco-Singapore",
    "link": "Click Here"
    }
]
Points to remember:

-Additional baggage allowance is not given on Air India flights operated with ATR/CRJ aircraft, airline partners’ flights, and Air India codeshare flights.
-Silver members are eligible for additional baggage allowance on flights operated by Air India only, whereas Gold and Platinum members (corresponding to Star Gold) are eligible for additional baggage allowance on Star Alliance flights as well.
-There is a ban on the carriage of lithium battery-powered self-balancing devices in cabin baggage. Power banks cannot be carried in checked baggage but in cabin baggage only.
-Samsung Galaxy Note 7 is banned and cannot be carried on flights in either baggage.
"""
prompt = ChatPromptTemplate.from_template(sys_template)
output_parser = StrOutputParser()

def get_response(input_text, chat_history):
    chain = prompt | llm | output_parser
    result = chain.invoke({"context": context, "chat_history": chat_history, "input_text": input_text})
    return result

st.title("Airline Chatbot Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("What is up?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    response = get_response(prompt_text, st.session_state.chat_history)
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})