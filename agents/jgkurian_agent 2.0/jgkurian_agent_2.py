import logging
import json
from random import randint
from time import time
from typing import cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from ..alex_agent.utils.opponent_model import OpponentModel


class JgkurianAgent(DefaultParty):
    """
    Jonathan's agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """
        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]
                self.load_data()

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Jonathan's Agent"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        if isinstance(action, Offer):
            # Create opponent model if not yet initialized
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # Update opponent model with bid
            self.opponent_model.update(bid)

            # Record utility of the bid
            if not hasattr(self, "opponent_bid_utilities"):
                self.opponent_bid_utilities = []
            self.opponent_bid_utilities.append(self.profile.getUtility(bid))

            # Record time of the bid (relative to progress)
            if not hasattr(self, "opponent_offer_times"):
                self.opponent_offer_times = []
            self.opponent_offer_times.append(self.progress.get(time() * 1000))

            # Record the bid itself for analysis
            if not hasattr(self, "opponent_bids"):
                self.opponent_bids = []
            self.opponent_bids.append(bid)

            # Set bid as last received
            self.last_received_bid = bid


    def my_turn(self):
        """Decide upon an action to perform and send it to the opponent."""
        # Track the utility of the last received bid (if available)
        if self.last_received_bid:
            if not hasattr(self, "received_bid_utilities"):
                self.received_bid_utilities = []
            self.received_bid_utilities.append(self.profile.getUtility(self.last_received_bid))

        # Decide whether to accept or offer a counter bid
        if self.accept_condition(self.last_received_bid):
            action = Accept(self.me, self.last_received_bid)
        else:
            bid = self.find_bid()
            if not hasattr(self, "proposed_bids"):
                self.proposed_bids = []
            self.proposed_bids.append(bid)
            action = Offer(self.me, bid)

        # Send the action
        self.send_action(action)
        
    def load_data(self):
        """Load saved data for learning - utilities, time, and bids."""
        try:
            # Construct the file path for the opponent's data
            file_path = f"{self.storage_dir}/opponent_data_{self.other}.json"

            # Attempt to open and read the saved data
            with open(file_path, "r") as f:
                data = json.load(f)

            # Load data into the corresponding attributes
            self.opponent_bid_utilities = data.get("opponent_bid_utilities", [])
            self.opponent_offer_times = data.get("opponent_offer_times", [])
            self.opponent_bids = [
                Bid.fromString(bid) for bid in data.get("opponent_bids", [])
            ]

            self.logger.log(logging.INFO, f"Loaded data for opponent: {self.other}")
        except FileNotFoundError:
            # If no file exists, initialize attributes to empty lists
            self.opponent_bid_utilities = []
            self.opponent_offer_times = []
            self.opponent_bids = []
            self.logger.log(
                logging.WARNING, f"No saved data found for opponent: {self.other}"
            )
        except Exception as e:
            # Log unexpected errors
            self.logger.log(logging.ERROR, f"Error loading data for opponent: {e}")


    def save_data(self):
        """Here, we save data for learning - we store utilities, time, and bids."""
        data = {
            "opponent_bid_utilities": getattr(self, "opponent_bid_utilities", []),
            "opponent_offer_times": getattr(self, "opponent_offer_times", []),
            "opponent_bids": [str(bid) for bid in getattr(self, "opponent_bids", [])],
        }

        with open(f"{self.storage_dir}/opponent_data_{self.other}.json", "w") as f:
            json.dump(data, f)


    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        progress = self.progress.get(time() * 1000)
        bid_utility = self.profile.getUtility(bid)

        # Dynamic utility thresholds based on opponent behavior
        if progress > 0.98:
            min_utility = 0.5  # Less aggressive at deadline
        elif progress > 0.95:
            min_utility = 0.8
        else:
            min_utility = 0.85

        return bid_utility > min_utility

    def find_bid(self) -> Bid:
            domain = self.profile.getDomain()
            all_bids = AllBidsList(domain)
            
            best_bid_score = 0.0
            best_bid = None

            # Adaptive sampling based on progress
            progress = self.progress.get(time() * 1000)
            base_attempts = 1000 if progress < 0.8 else 500
            
            # Store previously successful bids
            if not hasattr(self, 'successful_bids'):
                self.successful_bids = []
            
            # Try variations of previously successful bids first
            for prev_bid in self.successful_bids[-3:]:
                modified_bid = self.modify_bid(prev_bid, domain)
                if modified_bid:
                    bid_score = self.score_bid(modified_bid)
                    if bid_score > best_bid_score:
                        best_bid_score, best_bid = bid_score, modified_bid

            # take 500 attempts to find a bid according to a heuristic score
            for _ in range(base_attempts):
                bid = all_bids.get(randint(0, all_bids.size() - 1))
                bid_score = self.score_bid(bid)
                if bid_score > best_bid_score:
                    best_bid_score, best_bid = bid_score, bid
                    
            if best_bid and best_bid_score > 0.8:
                self.successful_bids.append(best_bid)
                if len(self.successful_bids) > 10:
                    self.successful_bids.pop(0)

            return best_bid

    def modify_bid(self, bid: Bid, domain) -> Bid:
        """Create a variation of a successful previous bid"""
        try:
            new_bid = bid.copy()
            issues = domain.getIssues()
            # Modify 1-2 random issues
            for _ in range(randint(1, 2)):
                issue = issues[randint(0, len(issues) - 1)]
                values = domain.getValues(issue)
                new_bid.setValue(issue, values[randint(0, len(values) - 1)])
            return new_bid
        except:
            return None

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.2) -> float:
        """Enhanced bid scoring that protects our utility while considering opponent"""
        progress = self.progress.get(time() * 1000)
        our_utility = float(self.profile.getUtility(bid))

        # Phase-based strategy
        if progress <= 0.4:
            adjusted_alpha = alpha
            adjusted_eps = eps
        elif 0.4 < progress <= 0.7:
            adjusted_alpha = min(alpha + 0.05, 0.95)
            adjusted_eps = eps * 1.2
        elif 0.7 < progress <= 0.9:
            adjusted_alpha = min(alpha + 0.1, 0.98)
            adjusted_eps = eps * 0.8
        else:
            adjusted_alpha = max(alpha - (progress - 0.9) * 1.5, 0.6)
            adjusted_eps = eps * 1.5

        # Base score calculation with time pressure
        time_pressure = 1.0 - (progress ** (1 / adjusted_eps))
        base_score = adjusted_alpha * time_pressure * our_utility

        # Opponent modeling with utility protection
        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            
            # Calculate opponent weight with limits
            if hasattr(self, "opponent_bid_utilities") and self.opponent_bid_utilities:
                recent_utilities = self.opponent_bid_utilities[-3:]
                avg_recent = sum(recent_utilities) / len(recent_utilities)
                # Limit opponent cooperation impact
                opponent_cooperation = min(float(avg_recent) / 0.8, 0.8)
                # Smaller opponent weight to protect our utility
                opponent_weight = min(0.3, (1.0 - adjusted_alpha * time_pressure)) * opponent_cooperation
            else:
                opponent_weight = min(0.3, 1.0 - adjusted_alpha * time_pressure)
            
            # Smaller balance bonus
            if our_utility >= 0.7:  # Only consider balance for good utilities
                balance_ratio = min(our_utility, opponent_utility) / max(our_utility, opponent_utility)
                balance_bonus = 0.05 * balance_ratio  # Linear scaling, smaller bonus
            else:
                balance_bonus = 0
                
            # Combine scores with utility protection
            opponent_score = opponent_weight * (opponent_utility + balance_bonus)
            final_score = base_score + (opponent_score * 0.5)  # Reduce opponent impact
            
            # Stronger bonus for high utility
            if our_utility > 0.85:
                final_score *= 1.2
            elif our_utility > 0.75:
                final_score *= 1.1
                
            # Ensure final score doesn't drop below base utility-driven score
            return max(final_score, our_utility * 0.9)

        return base_score
