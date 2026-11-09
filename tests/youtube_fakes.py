"""Stateful read/write API boundary fixture, never a publication verdict stub."""
import copy
from unittest.mock import Mock


def add_video_status_api(service, video_id, channel_id=None):
    state = {"privacyStatus": "private"}

    def listed(**kwargs):
        item = {"id": video_id, "status": copy.deepcopy(state),
                "snippet": {"channelId": channel_id},
                "processingDetails": {"processingStatus": "succeeded"}}
        return Mock(execute=Mock(return_value={"items": [item]}))

    def updated(**kwargs):
        def execute():
            state.clear()
            state.update(copy.deepcopy(kwargs["body"]["status"]))
            return {"id": video_id, "status": copy.deepcopy(state)}
        return Mock(execute=Mock(side_effect=execute))

    service.videos().list.side_effect = listed
    service.videos().update.side_effect = updated
    return state
