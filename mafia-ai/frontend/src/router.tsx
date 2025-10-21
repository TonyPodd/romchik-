// src/router.tsx
import { createBrowserRouter } from "react-router-dom";
import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { TableDetectionPage } from "./pages/TableDetectionPage";
import { FaceRegistrationPage } from "./pages/new/FaceRegistrationPage";
import { VoiceRegistrationPage } from "./pages/VoiceRegistrationPage";
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
        path: "setup/table",
        element: <TableDetectionPage />,
      },
      {
        path: "setup/players",
        element: <FaceRegistrationPage />,
      },
      {
        path: "setup/voice",
        element: <VoiceRegistrationPage />,
      },
      {
        path: "game/live",
        element: <GameLivePage />,
      },
      {
        path: "game/stats",
        element: <GameStatsPage />,
      },
    ],
  },
]);
