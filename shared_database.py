"""
Shared Database Manager for AI Chatbot and Resume Analyzer
Centralized database that stores all user interactions and personalization reports
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import hmac
import hashlib

WEBHOOK_EVENTS = [
    "resume.analyzed",
    "profile.updated", 
    "user.created",
    "chat.interaction"
]

def register_webhook(self, url: str, events: list, secret: str = None):
    webhooks = self._load_json("webhooks.json", default=[])
    webhook = {
        "id": str(uuid.uuid4()),
        "url": url,
        "events": events,
        "secret": secret,
        "created_at": datetime.now().isoformat(),
        "active": True
    }
    webhooks.append(webhook)
    self._save_json("webhooks.json", webhooks)
    return webhook

def get_webhooks_for_event(self, event: str):
    webhooks = self._load_json("webhooks.json", default=[])
    return [w for w in webhooks if event in w["events"] and w["active"]]

def deregister_webhook(self, webhook_id: str):
    webhooks = self._load_json("webhooks.json", default=[])
    webhooks = [w for w in webhooks if w["id"] != webhook_id]
    self._save_json("webhooks.json", webhooks)

class SharedDatabase:
    """
    Unified database manager for Academic Chatbot and Resume Analyzer.
    Stores all user data and personalization reports in a single location.
    """
    
    def __init__(self, storage_dir: str = "shared_data"):
        self.storage_dir = Path(storage_dir)
        self.users_file = self.storage_dir / "users.json"
        self.interactions_file = self.storage_dir / "interactions.json"
        self.profiles_file = self.storage_dir / "user_profiles.json"
        self.reports_file = self.storage_dir / "personalization_reports.json"
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize JSON files if they don't exist"""
        if not self.users_file.exists():
            self._save_json(self.users_file, {})
        
        if not self.interactions_file.exists():
            self._save_json(self.interactions_file, {})
        
        if not self.profiles_file.exists():
            self._save_json(self.profiles_file, {})
        
        if not self.reports_file.exists():
            self._save_json(self.reports_file, {})
    
    def _load_json(self, filepath: Path) -> dict:
        """Load JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return {}
    
    def _save_json(self, filepath: Path, data: dict):
        """Save JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
    
    # ============================
    # User Management
    # ============================
    
    def get_or_create_user(self, username: str) -> dict:
        """Get user or create if doesn't exist"""
        users = self._load_json(self.users_file)
        
        if username not in users:
            users[username] = {
                "username": username,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "modules_used": [],
                "session_ids": {
                    "chatbot": [],
                    "resume_analyzer": []
                },
                "total_interactions": 0
            }
            self._save_json(self.users_file, users)
            logger.info(f"Created new user: {username}")
        
        return users[username]
    
    def update_user_modules(self, username: str, module: str):
        """Update modules used by user"""
        users = self._load_json(self.users_file)
        
        if username in users:
            if module not in users[username]["modules_used"]:
                users[username]["modules_used"].append(module)
                users[username]["updated_at"] = datetime.now().isoformat()
                self._save_json(self.users_file, users)
                logger.info(f"Updated modules for {username}: added {module}")
                return True
        return False
    
    def add_session_to_user(self, username: str, module: str, session_id: str):
        """Add session_id to user's session list for specific module"""
        users = self._load_json(self.users_file)
        
        if username in users:
            if session_id not in users[username]["session_ids"][module]:
                users[username]["session_ids"][module].append(session_id)
                users[username]["total_interactions"] += 1
                users[username]["updated_at"] = datetime.now().isoformat()
                self.update_user_modules(username, module)
                self._save_json(self.users_file, users)
                return True
        return False
    
    def get_user_sessions(self, username: str, module: Optional[str] = None) -> List[str]:
        """Get all session IDs for a user, optionally filtered by module"""
        user = self.get_or_create_user(username)
        
        if module:
            return user["session_ids"].get(module, [])
        else:
            # Return all sessions from all modules
            all_sessions = []
            for sessions in user["session_ids"].values():
                all_sessions.extend(sessions)
            return all_sessions
    
    # ============================
    # Session/Interaction Management
    # ============================
    
    def save_interaction(self, username: str, module: str, session_id: str, interaction_data: dict):
        """
        Save or update an interaction (conversation or resume analysis)
        
        Args:
            username: User identifier
            module: "chatbot" or "resume_analyzer"
            session_id: Unique session identifier
            interaction_data: Data specific to the interaction
        """
        interactions = self._load_json(self.interactions_file)
        
        # Create composite key: username:module:session_id
        key = f"{username}:{module}:{session_id}"
        
        if key not in interactions:
            interactions[key] = {
                "username": username,
                "module": module,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "data": {}
            }
        
        # Update interaction
        interactions[key]["updated_at"] = datetime.now().isoformat()
        interactions[key]["data"] = interaction_data
        
        self._save_json(self.interactions_file, interactions)
        
        # Add session to user's list
        self.add_session_to_user(username, module, session_id)
        
        logger.info(f"Saved {module} interaction for {username} with session {session_id}")
    
    def get_interaction(self, username: str, module: str, session_id: str) -> Optional[dict]:
        """Get specific interaction"""
        interactions = self._load_json(self.interactions_file)
        key = f"{username}:{module}:{session_id}"
        return interactions.get(key, None)
    
    def get_user_interactions(self, username: str, module: Optional[str] = None) -> List[dict]:
        """
        Get all interactions for a user, optionally filtered by module
        
        Args:
            username: User identifier
            module: Optional filter - "chatbot", "resume_analyzer", or None for all
        
        Returns:
            List of interaction records
        """
        interactions = self._load_json(self.interactions_file)
        user_interactions = []
        
        for key, interaction in interactions.items():
            if interaction["username"] == username:
                if module is None or interaction["module"] == module:
                    user_interactions.append(interaction)
        
        # Sort by created_at (most recent first)
        user_interactions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return user_interactions
    
    def delete_interaction(self, username: str, module: str, session_id: str):
        """Delete an interaction"""
        interactions = self._load_json(self.interactions_file)
        users = self._load_json(self.users_file)
        
        # Remove from interactions
        key = f"{username}:{module}:{session_id}"
        if key in interactions:
            del interactions[key]
            self._save_json(self.interactions_file, interactions)
        
        # Remove from user's session list
        if username in users and session_id in users[username]["session_ids"][module]:
            users[username]["session_ids"][module].remove(session_id)
            users[username]["total_interactions"] = max(0, users[username]["total_interactions"] - 1)
            self._save_json(self.users_file, users)
    
    # ============================
    # Chatbot-specific methods
    # ============================
    
    def save_chatbot_conversation(self, username: str, chat_id: str, conversation_data: dict):
        """Save chatbot conversation"""
        self.save_interaction(username, "chatbot", chat_id, {
            "title": conversation_data.get("title", "New Conversation"),
            "messages": conversation_data.get("messages", []),
            "preferences": conversation_data.get("preferences", {}),
            "message_count": len(conversation_data.get("messages", [])),
            "last_message_at": datetime.now().isoformat()
        })
    
    def get_chatbot_conversation(self, username: str, chat_id: str) -> Optional[dict]:
        """Get specific chatbot conversation"""
        interaction = self.get_interaction(username, "chatbot", chat_id)
        return interaction["data"] if interaction else None
    
    def get_user_chatbot_conversations(self, username: str) -> List[dict]:
        """Get all chatbot conversations for a user"""
        interactions = self.get_user_interactions(username, "chatbot")
        
        conversations = []
        for interaction in interactions:
            conversations.append({
                "chat_id": interaction["session_id"],
                "username": username,
                "created_at": interaction["created_at"],
                "updated_at": interaction["updated_at"],
                "title": interaction["data"].get("title", ""),
                "messages": interaction["data"].get("messages", []),
                "preferences": interaction["data"].get("preferences", {}),
                "message_count": interaction["data"].get("message_count", 0)
            })
        
        return conversations
    
    # ============================
    # Resume Analyzer-specific methods
    # ============================
    
    def save_resume_analysis(self, username: str, analysis_id: str, analysis_data: dict):
        """Save resume analysis"""
        # Extract detailed insights from the analysis result
        analysis_result = analysis_data.get("analysis_result", {})
        
        # Get strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if "strengths_analysis" in analysis_result:
            strengths = [s.get("strength", "") for s in analysis_result["strengths_analysis"][:5]]
        
        if "weaknesses_analysis" in analysis_result:
            weaknesses = [w.get("weakness", "") for w in analysis_result["weaknesses_analysis"][:5]]
        
        # Get professional profile info
        professional_profile = analysis_result.get("executive_summary", {}).get("professional_profile", {})
        
        self.save_interaction(username, "resume_analyzer", analysis_id, {
            "target_role": analysis_data.get("target_role", ""),
            "overall_score": analysis_data.get("overall_score", 0),
            "recommendation_level": analysis_data.get("recommendation_level", ""),
            "analysis_result": analysis_result,
            "uploaded_at": analysis_data.get("uploaded_at", datetime.now().isoformat()),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "professional_profile": professional_profile,
            "technical_skills_count": professional_profile.get("technical_skills_count", 0),
            "experience_level": professional_profile.get("experience_level", ""),
            "achievement_metrics": professional_profile.get("achievement_metrics", "")
        })
    
    def get_resume_analysis(self, username: str, analysis_id: str) -> Optional[dict]:
        """Get specific resume analysis"""
        interaction = self.get_interaction(username, "resume_analyzer", analysis_id)
        return interaction["data"] if interaction else None
    
    def get_user_resume_analyses(self, username: str) -> List[dict]:
        """Get all resume analyses for a user"""
        interactions = self.get_user_interactions(username, "resume_analyzer")
        
        analyses = []
        for interaction in interactions:
            analyses.append({
                "analysis_id": interaction["session_id"],
                "username": username,
                "created_at": interaction["created_at"],
                "updated_at": interaction["updated_at"],
                "target_role": interaction["data"].get("target_role", ""),
                "overall_score": interaction["data"].get("overall_score", 0),
                "recommendation_level": interaction["data"].get("recommendation_level", ""),
                "analysis_result": interaction["data"].get("analysis_result", {}),
                "uploaded_at": interaction["data"].get("uploaded_at", ""),
                "strengths": interaction["data"].get("strengths", []),
                "weaknesses": interaction["data"].get("weaknesses", [])
            })
        
        return analyses
    
    # ============================
    # Personalization Profile & Report Management
    # ============================
    
    def save_user_profile(self, username: str, profile_data: dict):
        """Save user personality profile"""
        profiles = self._load_json(self.profiles_file)
        
        profiles[username] = {
            "username": username,
            "updated_at": datetime.now().isoformat(),
            "profile": profile_data
        }
        
        self._save_json(self.profiles_file, profiles)
        logger.info(f"Saved personality profile for {username}")
    
    def get_user_profile(self, username: str) -> Optional[dict]:
        """Get user personality profile"""
        profiles = self._load_json(self.profiles_file)
        return profiles.get(username, {}).get("profile")
    
    def save_personalization_report(self, username: str, report_data: dict):
        """Save personalization report"""
        reports = self._load_json(self.reports_file)
        
        report_id = report_data.get("report_id", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if username not in reports:
            reports[username] = []
        
        # Add new report to the beginning of the list
        reports[username].insert(0, {
            "report_id": report_id,
            "generated_at": datetime.now().isoformat(),
            "report": report_data
        })
        
        # Keep only the last 10 reports per user
        reports[username] = reports[username][:10]
        
        self._save_json(self.reports_file, reports)
    
    def get_latest_personalization_report(self, username: str) -> Optional[dict]:
        """Get the most recent personalization report for a user"""
        reports = self._load_json(self.reports_file)
        user_reports = reports.get(username, [])
        
        if user_reports:
            return user_reports[0]["report"]
        return None
    
    def get_all_personalization_reports(self, username: str) -> List[dict]:
        """Get all personalization reports for a user"""
        reports = self._load_json(self.reports_file)
        return reports.get(username, [])
    
    # ============================
    # Enhanced Resume Insights for Personalization
    # ============================
    
    def get_resume_insights(self, username: str) -> Dict[str, Any]:
        """Get aggregated resume analysis insights"""
        resume_analyses = self.get_user_resume_analyses(username)
        
        if not resume_analyses:
            return {
                "total_analyses": 0,
                "average_score": 0,
                "target_roles": [],
                "improvement_trend": "No data",
                "common_strengths": [],
                "common_weaknesses": [],
                "technical_skills_trend": 0,
                "experience_levels": []
            }
        
        total_analyses = len(resume_analyses)
        scores = [a.get("overall_score", 0) for a in resume_analyses]
        average_score = sum(scores) / len(scores) if scores else 0
        target_roles = list(set([a.get("target_role", "") for a in resume_analyses if a.get("target_role")]))
        
        # Collect strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        for analysis in resume_analyses:
            all_strengths.extend(analysis.get("strengths", []))
            all_weaknesses.extend(analysis.get("weaknesses", []))
        
        # Count frequencies
        from collections import Counter
        strength_counter = Counter(all_strengths)
        weakness_counter = Counter(all_weaknesses)
        
        # Calculate improvement trend
        if len(scores) >= 2:
            if scores[0] > scores[-1]:
                trend = "Improving"
            elif scores[0] < scores[-1]:
                trend = "Declining"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient data"
        
        return {
            "total_analyses": total_analyses,
            "average_score": round(average_score, 1),
            "latest_score": scores[0] if scores else 0,
            "target_roles": target_roles,
            "improvement_trend": trend,
            "common_strengths": [s for s, _ in strength_counter.most_common(5)],
            "common_weaknesses": [w for w, _ in weakness_counter.most_common(5)],
            "technical_skills_trend": sum([a.get("technical_skills_count", 0) for a in resume_analyses]) / total_analyses,
            "experience_levels": [a.get("experience_level", "") for a in resume_analyses],
            "analyses_history": [
                {
                    "date": a.get("created_at", ""),
                    "role": a.get("target_role", ""),
                    "score": a.get("overall_score", 0)
                }
                for a in resume_analyses[:5]
            ]
        }
    
    # ============================
    # Export for Personalization Module
    # ============================
    
    def export_user_data_for_personalization(self, username: str) -> Dict[str, Any]:
        """
        Export all user data in format suitable for personalization module
        
        Returns:
            Dictionary with chatbot and resume analyzer data
        """
        user = self.get_or_create_user(username)
        chatbot_interactions = self.get_user_interactions(username, "chatbot")
        resume_analyses = self.get_user_interactions(username, "resume_analyzer")
        
        # Extract all messages for analysis
        all_messages = []
        for interaction in chatbot_interactions:
            messages = interaction["data"].get("messages", [])
            for msg in messages:
                all_messages.append({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "is_recommendation": msg.get("is_recommendation", False)
                })
        
        # Get resume insights
        resume_insights = self.get_resume_insights(username)
        
        return {
            "username": username,
            "user_info": user,
            "all_messages": all_messages,
            "chatbot_interactions": chatbot_interactions,
            "resume_analyses": resume_analyses,
            "resume_insights": resume_insights,
            "existing_profile": self.get_user_profile(username),
            "latest_report": self.get_latest_personalization_report(username),
            "total_messages": len(all_messages),
            "total_analyses": len(resume_analyses),
            "modules_used": user.get("modules_used", [])
        }
    
    # ============================
    # Statistics and Utility
    # ============================
    
    def get_user_stats(self, username: str) -> Dict[str, Any]:
        """Get user statistics"""
        user = self.get_or_create_user(username)
        chatbot_sessions = len(self.get_user_sessions(username, "chatbot"))
        resume_sessions = len(self.get_user_sessions(username, "resume_analyzer"))
        resume_insights = self.get_resume_insights(username)
        
        return {
            "username": username,
            "created_at": user["created_at"],
            "updated_at": user.get("updated_at", user["created_at"]),
            "modules_used": user.get("modules_used", []),
            "total_sessions": chatbot_sessions + resume_sessions,
            "chatbot_sessions": chatbot_sessions,
            "resume_analyzer_sessions": resume_sessions,
            "total_interactions": user.get("total_interactions", 0),
            "has_personalization_profile": self.get_user_profile(username) is not None,
            "latest_report_date": self.get_latest_personalization_report(username).get("generated_at") if self.get_latest_personalization_report(username) else None,
            "resume_insights": resume_insights
        }
    
    def get_all_users(self) -> List[str]:
        """Get list of all usernames"""
        users = self._load_json(self.users_file)
        return list(users.keys())
