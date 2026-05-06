use anyhow::Result;
use nostr_sdk::prelude::*;

/// Shared nostr client for relay connections and on-demand event fetches.
#[derive(Clone)]
pub struct NostrClient {
    client: Client,
}

impl NostrClient {
    pub async fn new(relays: &[String], nsec: Option<&str>) -> Result<Self> {
        let mut builder = Client::builder();

        let keys = match nsec {
            Some(nsec_str) if !nsec_str.is_empty() => {
                let secret_key = SecretKey::from_bech32(nsec_str)
                    .or_else(|_| SecretKey::from_hex(nsec_str))?;
                Keys::new(secret_key)
            }
            _ => Keys::generate(),
        };
        builder = builder.signer(keys);

        let client = builder.build();

        for relay_url in relays {
            client.add_relay(relay_url).await?;
        }

        Ok(Self { client })
    }

    pub async fn connect(&self) {
        self.client.connect().await;
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Fetch a single event by ID from relays, return it (no caching here).
    pub async fn fetch_event_by_id(&self, event_id: &str) -> Result<Option<nostr_sdk::Event>> {
        let id = EventId::from_hex(event_id)?;
        let filter = Filter::new().id(id).limit(1);

        let events = self.client
            .fetch_events(filter, std::time::Duration::from_secs(10))
            .await?;

        Ok(events.into_iter().next())
    }

    /// Count how many kind 3 (Contacts) events reference this pubkey in a 'p' tag.
    /// This gives a rough follower count from the relays we're connected to.
    pub async fn fetch_follower_count(&self, pubkey: &str, timeout_secs: u64) -> Result<usize> {
        let pk = PublicKey::from_hex(pubkey)?;
        let filter = Filter::new()
            .kind(Kind::ContactList)
            .pubkey(pk);

        let events = self.client
            .fetch_events(filter, std::time::Duration::from_secs(timeout_secs))
            .await?;

        Ok(events.len())
    }
}
