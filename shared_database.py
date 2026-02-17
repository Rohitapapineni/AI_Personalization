"""
Shared Database Manager for AI Chatbot and Resume Analyzer
Centralized database that stores all user interactions and personalization reports
"""

import json
import uuid
import hmac
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

WEBHOOK_EVENTS = [
    "resume.analyzed",
    "profile.updated",
    "user.created",
    "chat.interaction"
]


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
        self.webhooks_file = self.storage_dir / "webhooks.json"

        self.storage_dir.mkdir(exist_ok=True)
        self._initialize_files()

    def _initialize_files(self):
        """Initialize JSON files if they don't exist"""
        for filepath, default in [
            (self.users_file, {}),
            (self.interactions_file, {}),
            (self.profiles_file, {}),
            (self.reports_file, {}),
            (self.webhooks_file, []),
        ]:
            if not filepath.exists():
                self._save_json(filepath, default)

    def _load_json(self, filepath: Path):
        """Load JSON file — accepts a Path object"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            # Return correct empty type based on file
            return [] if filepath == self.webhooks_file else {}

    def _save_json(self, filepath: Path, data):
        """Save JSON file — accepts a Path object"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")

    # ============================
    # Webhook Management  ← NOW PROPERLY INSIDE THE CLASS
    # ============================

    def register_webhook(self, url: str, events: list, secret: str = None) -> dict:
        """Register a new webhook"""
        webhooks = self._load_json(self.webhooks_file)
        webhook = {
            "id": str(uuid.uuid4()),
            "url": url,
            "events": events,
            "secret": secret,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        webhooks.append(webhook)
        self._save_json(self.webhooks_file, webhooks)
        logger.info(f"Registered webhook: {url} for events {events}")
        return webhook

    def get_webhooks_for_event(self, event: str) -> list:
        """Get all active webhooks registered for a given event"""
        webhooks = self._load_json(self.webhooks_file)
        return [w for w in webhooks if event in w.get("events", []) and w.get("active", False)]

    def deregister_webhook(self, webhook_id: str):
        """Remove a webhook by ID"""
        webhooks = self._load_json(self.webhooks_file)
        webhooks = [w for w in webhooks if w["id"] != webhook_id]
        self._save_json(self.webhooks_file, webhooks)
        logger.info(f"Deregistered webhook: {webhook_id}")

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
                return True
        return False

    def add_session_to_user(self, username: str, module: str, session_id: str):
        """Add session_id to user's session list for specific module"""
        users = self._load_json(self.users_file)

        if username in users:
            if session_id not in users[username]["session_ids"].get(module, []):
                if module not in users[username]["session_ids"]:
                    users[username]["session_ids"][module] = []
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
            all_sessions = []
            for sessions in user["session_ids"].values():
                all_sessions.extend(sessions)
            return all_sessions

    # ============================
    # Session / Interaction Management
    # ============================

    def save_interaction(self, username: str, module: str, session_id: str, interaction_data: dict):
        """Save or update an interaction"""
        interactions = self._load_json(self.interactions_file)

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

        interactions[key]["updated_at"] = datetime.now().isoformat()
        interactions[key]["data"] = interaction_data

        self._save_json(self.interactions_file, interactions)
        self.add_session_to_user(username, module, session_id)

        logger.info(f"Saved {module} interaction for {username} (session: {session_id})")

    def get_interaction(self, username: str, module: str, session_id: str) -> Optional[dict]:
        """Get specific interaction"""
        interactions = self._load_json(self.interactions_file)
        key = f"{username}:{module}:{session_id}"
        return interactions.get(key, None)

    def get_user_interactions(self, username: str, module: Optional[str] = None) -> List[dict]:
        """Get all interactions for a user, optionally filtered by module"""
        interactions = self._load_json(self.interactions_file)
        user_interactions = []

        for key, interaction in interactions.items():
            if interaction["username"] == username:
                if module is None or interaction["module"] == module:
                    user_interactions.append(interaction)

        user_interactions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return user_interactions

    def delete_interaction(self, username: str, module: str, session_id: str):
        """Delete an interaction"""
        interactions = self._load_json(self.interactions_file)
        users = self._load_json(self.users_file)

        key = f"{username}:{module}:{session_id}"
        if key in interactions:
            del interactions[key]
            self._save_json(self.interactions_file, interactions)

        if username in users:
            sessions = users[username]["session_ids"].get(module, [])
            if session_id in sessions:
                sessions.remove(session_id)
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
        """Save resume analysis — extracts strengths/weaknesses for cross-module sharing"""
        analysis_result = analysis_data.get("analysis_result", {})

        strengths = []
        weaknesses = []

        if "strengths_analysis" in analysis_result:
            strengths = [s.get("strength", "") for s in analysis_result["strengths_analysis"][:5]]

        if "weaknesses_analysis" in analysis_result:
            weaknesses = [w.get("weakness", "") for w in analysis_result["weaknesses_analysis"][:5]]

        professional_profile = (
            analysis_result
            .get("executive_summary", {})
            .get("professional_profile", {})
        )

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

    def delete_user_profile(self, username: str):
        """Delete user profile so it gets regenerated fresh on next request"""
        profiles = self._load_json(self.profiles_file)
        if username in profiles:
            del profiles[username]
            self._save_json(self.profiles_file, profiles)
            logger.info(f"Deleted profile for {username} (will regenerate fresh)")

    def save_personalization_report(self, username: str, report_data: dict):
        """Save personalization report"""
        reports = self._load_json(self.reports_file)

        report_id = report_data.get("report_id", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if username not in reports:
            reports[username] = []

        reports[username].insert(0, {
            "report_id": report_id,
            "generated_at": datetime.now().isoformat(),
            "report": report_data
        })

        reports[username] = reports[username][:10]
        self._save_json(self.reports_file, reports)

    def get_latest_personalization_report(self, username: str) -> Optional[dict]:
        """Get the most recent personalization report for a user"""
        reports = self._load_json(self.reports_file)
        user_reports = reports.get(username, [])
        return user_reports[0]["report"] if user_reports else None

    def get_all_personalization_reports(self, username: str) -> List[dict]:
        """Get all personalization reports for a user"""
        reports = self._load_json(self.reports_file)
        return reports.get(username, [])

    # ============================
    # Resume Insights (for cross-module sharing)
    # ============================

    def get_resume_insights(self, username: str) -> Dict[str, Any]:
        """Get aggregated resume insights — used by both chatbot and personalization module"""
        resume_analyses = self.get_user_resume_analyses(username)

        if not resume_analyses:
            return {
                "total_analyses": 0,
                "average_score": 0,
                "latest_score": 0,
                "target_roles": [],
                "improvement_trend": "No data",
                "common_strengths": [],
                "common_weaknesses": [],
                "technical_skills_trend": 0,
                "experience_levels": [],
                "analyses_history": []
            }

        from collections import Counter

        scores = [a.get("overall_score", 0) for a in resume_analyses]
        average_score = sum(scores) / len(scores) if scores else 0
        target_roles = list(set([
            a.get("target_role", "") for a in resume_analyses
            if a.get("target_role") and a.get("target_role") != "general position"
        ]))

        all_strengths = []
        all_weaknesses = []
        for a in resume_analyses:
            all_strengths.extend(a.get("strengths", []))
            all_weaknesses.extend(a.get("weaknesses", []))

        if len(scores) >= 2:
            trend = "Improving" if scores[0] > scores[-1] else "Declining" if scores[0] < scores[-1] else "Stable"
        else:
            trend = "Insufficient data"

        strength_counter = Counter(all_strengths)
        weakness_counter = Counter(all_weaknesses)

        return {
            "total_analyses": len(resume_analyses),
            "average_score": round(average_score, 1),
            "latest_score": scores[0] if scores else 0,
            "target_roles": target_roles,
            "improvement_trend": trend,
            "common_strengths": [s for s, _ in strength_counter.most_common(5)],
            "common_weaknesses": [w for w, _ in weakness_counter.most_common(5)],
            "technical_skills_trend": sum([
                a.get("technical_skills_count", 0) for a in resume_analyses
            ]) / len(resume_analyses),
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
    # Full Export for Personalization Module
    # ============================

    def export_user_data_for_personalization(self, username: str) -> Dict[str, Any]:
        """Export all user data in format suitable for personalization module"""
        user = self.get_or_create_user(username)
        chatbot_interactions = self.get_user_interactions(username, "chatbot")
        resume_analyses = self.get_user_interactions(username, "resume_analyzer")

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
        latest_report = self.get_latest_personalization_report(username)

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
            "latest_report_date": latest_report.get("generated_at") if latest_report else None,
            "resume_insights": resume_insights
        }

    def get_all_users(self) -> List[str]:
        """Get list of all usernames"""
        users = self._load_json(self.users_file)
        return list(users.keys())
