import { createBrowserRouter } from "react-router-dom";
import { WelcomePage } from "./pages/new/WelcomePage";
import { PlayerCountPage } from "./pages/new/PlayerCountPage";
import { FaceRegistrationPage } from "./pages/new/FaceRegistrationPage";
import { VoiceRegistrationPage } from "./pages/new/VoiceRegistrationPage";
import { TableDetectionPage } from "./pages/new/TableDetectionPage";
import { GameProcessPage } from "./pages/new/GameProcessPage";
import { LogsSpeechPage } from "./pages/new/LogsSpeechPage";

export const routerNew = createBrowserRouter([
  {
    path: "/",
    element: <WelcomePage />,
  },
  {
    path: "/setup/players",
    element: <PlayerCountPage />,
  },
  {
    path: "/setup/face-registration",
    element: <FaceRegistrationPage />,
  },
  {
    path: "/setup/voice-registration",
    element: <VoiceRegistrationPage />,
  },
  {
    path: "/setup/voice",
    element: <VoiceRegistrationPage />,
  },
  {
    path: "/setup/table-detection",
    element: <TableDetectionPage />,
  },
  {
    path: "/game/process",
    element: <GameProcessPage />,
  },
  {
    path: "/logs_speech",
    element: <LogsSpeechPage />,
  },
]);
