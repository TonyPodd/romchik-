// src/router.tsx
import { createBrowserRouter } from "react-router-dom";
import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { SetupWizard } from "./pages/SetupWizard";
import { TableDetectionPage } from "./pages/TableDetectionPage";
import { PlayerEnrollmentPage } from "./pages/PlayerEnrollmentPage";
import { VoiceRegistrationPage } from "./pages/VoiceRegistrationPage";
import { GameSetupPage } from "./pages/GameSetupPage";
import { GameLivePage } from "./pages/GameLivePage";
import { GameStatsPage } from "./pages/GameStatsPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      {
        index: true,
        element: <HomePage />,
      },
      {
        path: "setup",
        element: <SetupWizard />,
        children: [
          {
            path: "table",
            element: <TableDetectionPage />,
          },
          {
            path: "players",
            element: <PlayerEnrollmentPage />,
          },
          {
            path: "voice",
            element: <VoiceRegistrationPage />,
          },
        ],
      },
      {
        path: "game",
        children: [
          {
            path: "setup",
            element: <GameSetupPage />,
          },
          {
            path: "live",
            element: <GameLivePage />,
          },
          {
            path: "stats",
            element: <GameStatsPage />,
          },
        ],
      },
    ],
  },
]);
